import asyncio
import base64
import dataclasses
import io
import logging
import os
import signal
import time
import uuid
from http import HTTPStatus
from typing import Any

import fastapi
import numpy as np
import psutil
import requests
import setproctitle
from transformers import AutoImageProcessor

from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.multimodal.manager.io_struct import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    DataType,
    GenerateMMReqInput,
    GenerateOpenAIAudioInput,
    TokenizedGenerateMMReqInput,
)
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    dataclass_to_string_truncated,
    kill_itself_when_parent_died,
)
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MMReqState:
    """Store the state of a request."""

    rid: str
    out_list: list[dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateMMReqInput
    created_time: float


class MultimodalTokenizer(TokenizerManager):
    """Tokenization manager for multimodal requests.

    `MultimodalTokenizer` accepts high-level multimodal generation requests
    (`GenerateMMReqInput`), tokenizes text inputs (and prepares image
    references when supported), forwards tokenized requests to the
    scheduler pipeline, and waits for/streams back results. It tracks the
    state of outstanding requests via `MMReqState` and uses a
    `TypeBasedDispatcher` to handle results arriving from the pipeline.
    """

    def __init__(self, server_args, port_args):
        """Initialize tokenizer, processor and result dispatcher.

        Loads processors for different modalities, initializes an in-memory
        map `rid_to_state` to track request state objects, and prepares a
        result dispatcher that routes batches of outputs back to
        `_handle_batch_output`.
        """
        super().__init__(server_args, port_args)
        self.wait_timeout = int(os.environ.get("SGLANG_WAIT_TIMEOUT", "600"))
        
        # Initialize processors for vision and audio
        self.image_processor = None
        self._init_image_processor(server_args.model_path)

        # Initialize audio processor (WhisperFeatureExtractor) for audio models
        self.audio_processor = None
        self.audio_config = {}
        self._init_audio_processor(server_args.model_path)

        self.rid_to_state: dict[str, MMReqState] = {}
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (list),
                    self._handle_batch_output,
                ),
                (
                    AbortReq,
                    self._handle_abort_req,
                ),
            ]
        )

    def _init_image_processor(self, model_path: str):
        """Initialize image processor for multimodal models."""
        try:
            # Use slow image processor to avoid torchvision dependency warning
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_path, use_fast=False
            )
        except Exception:
            logger.warning("Failed to load image processor from %s", model_path)

    def _init_audio_processor(self, model_path: str):
        """Initialize audio processor for audio models using transformers audio_utils.

        This loads the audio config and initializes mel filter bank and window function
        that match the official MiMo Audio implementation (power=1.0, log_mel="log").
        """
        import json
        import os

        model_paths_to_try = [model_path]

        # Load stage config to find audio encoder model path
        try:
            from sgl_jax.srt.multimodal.models.static_configs import get_stage_config_path
            from sgl_jax.srt.multimodal.manager.utils import load_stage_configs_from_yaml

            stage_config_path = get_stage_config_path(model_path)
            stage_configs = load_stage_configs_from_yaml(stage_config_path)

            for stage_cfg in stage_configs:
                scheduler = getattr(stage_cfg, 'scheduler', None)
                if scheduler == "audio_encoder":
                    audio_model_path = getattr(stage_cfg, 'model_path', None)
                    if audio_model_path and audio_model_path not in model_paths_to_try:
                        model_paths_to_try.insert(0, audio_model_path)
                    break
        except Exception as e:
            logger.warning("Could not load stage config: %s", e)

        # Fallback for MiMo Audio models
        if "mimo" in model_path.lower() and "audio" in model_path.lower():
            if "XiaomiMiMo/MiMo-Audio-Tokenizer" not in model_paths_to_try:
                model_paths_to_try.append("XiaomiMiMo/MiMo-Audio-Tokenizer")

        logger.warning("Attempting to init audio processor from paths: %s", model_paths_to_try)

        for try_path in model_paths_to_try:
            try:
                import huggingface_hub
                if os.path.isdir(try_path):
                    config_path = os.path.join(try_path, "config.json")
                else:
                    try:
                        config_path = huggingface_hub.hf_hub_download(try_path, "config.json", local_files_only=True)
                    except Exception:
                        config_path = huggingface_hub.hf_hub_download(try_path, "config.json", local_files_only=False)

                with open(config_path, "r") as f:
                    config = json.load(f)

                if "n_mels" in config and "sampling_rate" in config:
                    self.audio_config = config
                    # Use transformers audio_utils (no torch dependency, matches official MiMo)
                    from transformers.audio_utils import mel_filter_bank, window_function

                    # Get parameters from config (with MiMo Audio defaults)
                    sample_rate = config.get("sampling_rate", 24000)
                    n_fft = config.get("nfft", 960)
                    hop_length = config.get("hop_length", 240)
                    win_length = config.get("window_size", 960)
                    f_min = config.get("fmin", 0)
                    f_max = config.get("fmax", sample_rate // 2)  # Default to Nyquist
                    n_mels = config.get("n_mels", 128)

                    # Create mel filter bank
                    # Use HTK mel scale and no norm to match torchaudio defaults
                    self.mel_filters = mel_filter_bank(
                        num_frequency_bins=n_fft // 2 + 1,
                        num_mel_filters=n_mels,
                        min_frequency=f_min,
                        max_frequency=f_max,
                        sampling_rate=sample_rate,
                        norm=None,  # Match torchaudio default (no area normalization)
                        mel_scale="htk",  # Match torchaudio default
                    )

                    # Create window function
                    self.window = window_function(win_length, "hann")

                    # Store parameters for spectrogram computation
                    self.mel_params = {
                        "sample_rate": sample_rate,
                        "n_fft": n_fft,
                        "hop_length": hop_length,
                        "win_length": win_length,
                    }

                    # Store sampling rate for resampling
                    self.audio_processor = type('AudioProcessor', (), {'sampling_rate': sample_rate})()
                    logger.warning(
                        "Initialized transformers audio_utils from %s: sr=%d, n_fft=%d, hop=%d, n_mels=%d",
                        try_path, sample_rate, n_fft, hop_length, n_mels
                    )
                    return
            except Exception as e:
                logger.warning("Failed to init audio processor from %s: %s", try_path, e)
                import traceback
                logger.warning("Traceback: %s", traceback.format_exc())

        # Fallback: initialize with default MiMo Audio parameters
        logger.warning("Could not load config, using default MiMo Audio parameters")
        self._init_default_mel_processor()

    def _init_default_mel_processor(self):
        """Initialize mel processor with default MiMo Audio parameters.

        Uses the official MiMo Audio parameters:
        - sample_rate: 24000
        - n_fft: 960
        - hop_length: 240
        - win_length: 960
        - f_min: 0
        - f_max: 12000 (Nyquist)
        - n_mels: 128
        """
        from transformers.audio_utils import mel_filter_bank, window_function

        # Default MiMo Audio parameters
        sample_rate = 24000
        n_fft = 960
        hop_length = 240
        win_length = 960
        f_min = 0
        f_max = 12000  # Nyquist frequency
        n_mels = 128

        # Create mel filter bank
        # Use HTK mel scale and no norm to match torchaudio defaults
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1,
            num_mel_filters=n_mels,
            min_frequency=f_min,
            max_frequency=f_max,
            sampling_rate=sample_rate,
            norm=None,  # Match torchaudio default (no area normalization)
            mel_scale="htk",  # Match torchaudio default
        )

        # Create window function
        self.window = window_function(win_length, "hann")

        # Store parameters for spectrogram computation
        self.mel_params = {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
        }

        # Store sampling rate for resampling
        self.audio_processor = type('AudioProcessor', (), {'sampling_rate': sample_rate})()
        logger.warning(
            "Initialized transformers audio_utils with defaults: sr=%d, n_fft=%d, hop=%d, n_mels=%d",
            sample_rate, n_fft, hop_length, n_mels
        )

    def _preprocess_audio_to_mel(self, audio_array: np.ndarray, input_sr: int = None) -> tuple:
        """Convert raw audio waveform to mel spectrogram using transformers audio_utils.

        This matches the official MiMo Audio implementation:
        - Uses power=1.0 (amplitude spectrogram)
        - Applies natural log via log_mel="log"
        - Returns mel spectrogram in [batch, time, n_mels] format

        Args:
            audio_array: Raw audio waveform as numpy array, shape (samples,).
            input_sr: Input audio sample rate. If different from target rate, will resample.

        Returns:
            Tuple of (mel_spectrogram, input_lengths) as numpy arrays.
            mel_spectrogram shape: [batch, time, n_mels]
        """
        from transformers.audio_utils import spectrogram

        if not hasattr(self, 'mel_filters') or self.mel_filters is None:
            raise ValueError("Mel filter bank not initialized. Cannot preprocess audio.")

        # Ensure 1D array
        if audio_array.ndim == 2:
            audio_array = audio_array.squeeze(0)

        # Resample if input sample rate differs from target rate
        target_sr = self.audio_processor.sampling_rate
        if input_sr is not None and input_sr != target_sr:
            audio_array = self._resample_audio(audio_array, input_sr, target_sr)

        # Compute mel spectrogram with power=1.0 (matches official MiMo)
        # spectrogram() returns [n_mels, time] with log_mel applied
        mels = spectrogram(
            waveform=audio_array,
            window=self.window,
            frame_length=self.mel_params["n_fft"],
            hop_length=self.mel_params["hop_length"],
            fft_length=self.mel_params["n_fft"],
            power=1.0,  # Amplitude spectrogram (matches official MiMo)
            center=True,
            mel_filters=self.mel_filters,
            log_mel="log",  # Natural logarithm (matches official torch.log)
            mel_floor=1e-7,  # Matches official torch.clip(spec, min=1e-7)
        )

        # mels is [n_mels, time], transpose to [1, time, n_mels] for model input
        mels = mels.T[None, :, :]  # [1, time, n_mels]
        input_lens = np.array([mels.shape[1]])

        logger.info(
            "Audio preprocessing (transformers audio_utils): input_samples=%d, mel_shape=%s, input_lens=%s",
            len(audio_array),
            mels.shape,
            input_lens,
        )
        logger.info(
            "  Mel stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            mels.min(), mels.max(), mels.mean(), mels.std()
        )

        return mels, input_lens

    def _handle_batch_output(self, reqs: list):
        """Handle a batch of outputs returned from the pipeline.

        Marks the corresponding `MMReqState` as finished, sets its event to
        wake any waiters, and stores a simple success meta record. If a
        result arrives for an unknown `rid` it logs a warning.
        """
        if len(reqs) > 0 and self.server_args.log_requests:
            logger.info("handle_batch_output %s, self.rid_to_state %s", reqs, self.rid_to_state)
        for req in reqs:
            if req.rid in self.rid_to_state:
                self.rid_to_state[req.rid].finished = True
                self.rid_to_state[req.rid].event.set()

                out_data = {"success": True, "meta_info": {}}
                if hasattr(req, "audio_mode") and req.audio_mode is not None:
                    if req.audio_mode in ("asr", "audio_understanding") and req.generated_text_tokens is not None:
                        # Decode generated text tokens to string
                        tokens = req.generated_text_tokens
                        if hasattr(tokens, "tolist"):
                            tokens = tokens.tolist()
                        
                        # Store raw tokens for usage calculation
                        out_data["generated_text_tokens"] = tokens
                        
                        logger.info("ASR generated tokens: %s", tokens)
                        
                        if self.tokenizer:
                            # Use tokenizer to decode
                            decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                            if not decoded_text and tokens:
                                # Debug info if decode results in empty string
                                debug_tokens = tokens[:20]
                                decoded_text = f"[DEBUG: Empty Decode. Tokens: {debug_tokens}...]"
                            out_data["text"] = decoded_text
                            logger.info("ASR decoded text: '%s'", out_data["text"])
                        else:
                            # Fallback if tokenizer not available (unlikely)
                            out_data["text"] = str(tokens)
                            logger.warning("Tokenizer not initialized, returning raw tokens for ASR")

                self.rid_to_state[req.rid].out_list = [out_data]
            else:
                logger.warning(
                    "Received result for unknown request rid=%s. Known rids: %s",
                    req.rid,
                    list(self.rid_to_state.keys()),
                )

    def _handle_abort_req(self, recv_obj: AbortReq):
        """Handle an AbortReq returned from the scheduler.

        When a request is aborted (e.g., removed from the scheduler's queue
        before processing started), the scheduler sends an AbortReq back to
        notify the tokenizer. This method marks the request as finished with
        an abort status and wakes any waiting coroutines.
        """
        if recv_obj.rid not in self.rid_to_state:
            logger.warning(
                "Received abort for unknown request rid=%s. Known rids: %s",
                recv_obj.rid,
                list(self.rid_to_state.keys()),
            )
            return

        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.out_list.append(
            {
                "success": False,
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": recv_obj.aborted_message or "Request aborted",
                        "status_code": HTTPStatus.BAD_REQUEST,
                    },
                },
            }
        )
        state.event.set()
        logger.info("Abort completed for rid=%s", recv_obj.rid)

    async def generate_request(
        self,
        obj: GenerateMMReqInput,
        request: fastapi.Request | None = None,
    ):
        """High level API: accept a generation request and stream responses.

        This coroutine tokenizes the input (text and optional image refs),
        sends the tokenized request to the scheduler pipeline, and then
        asynchronously yields results as they arrive (supporting streaming
        if `obj.stream` is True). It respects client disconnects and a
        configured wait timeout.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )

        tokenized_obj = await self._tokenize_one_request(obj)
        state = self._send_one_request(obj, tokenized_obj, created_time)
        async for response in self._wait_one_response(obj, state, request):
            yield response

    async def _tokenize_one_request(self, obj: GenerateMMReqInput):
        """
        Converts text fields to token ids using the configured tokenizer.
        Image preprocessing / references are noted as TODO; when provided
        `input_ids` are passed through unchanged.
        """
        # Support both 'prompt' (multimodal) and 'text' (text-only) fields
        input_text = getattr(obj, "prompt", None) or getattr(obj, "text", None)
        neg_input_text = getattr(obj, "neg_prompt", None) or getattr(obj, "text", None)
        input_ids = getattr(obj, "input_ids", None)
        neg_input_ids = getattr(obj, "neg_input_ids", None)
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        if neg_input_ids is None and neg_input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but neg_input_text requires tokenization"
                )
            encoded = self.tokenizer(neg_input_text)
            neg_input_ids = encoded["input_ids"]
        if getattr(obj, "input_reference", None) is not None:
            # TODO: Handle image preprocessing for multimodal inputs
            pass

        return self._create_tokenized_object(
            obj, input_text, input_ids, neg_input_text, neg_input_ids
        )

    def _create_tokenized_object(
        self, obj: GenerateMMReqInput, input_text, input_ids, neg_input_text, neg_input_ids
    ):
        """Build `TokenizedGenerateMMReqInput` from the original request.

        Ensures a request id (`rid`) exists, and copies over relevant
        properties such as size, num_frames, data type and save_output flag.
        """
        rid = getattr(obj, "rid", None)
        if rid is None:
            rid = uuid.uuid4().hex

        tokenized_obj = TokenizedGenerateMMReqInput(
            rid=rid,
            prompt=input_text,
            negative_prompt=neg_input_text,
            input_ids=input_ids,
            negative_input_ids=neg_input_ids,
            size=getattr(obj, "size", None),
            num_frames=getattr(obj, "num_frames", None),
            num_inference_steps=getattr(obj, "num_inference_steps", 50),
            data_type=getattr(obj, "data_type", None),
            save_output=getattr(obj, "save_output", True),
        )
        return tokenized_obj

    def _send_one_request(
        self,
        obj: GenerateMMReqInput,
        tokenized_obj: TokenizedGenerateMMReqInput,
        created_time: float | None = None,
    ):
        """Send a tokenized request into the scheduling pipeline and track it.

        Constructs an `MMReqState` to wait for results and stores it in
        `rid_to_state` keyed by the request id.
        """
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = MMReqState(
            rid=tokenized_obj.rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[tokenized_obj.rid] = state
        return state

    async def _wait_one_response(
        self,
        obj: GenerateMMReqInput,
        state: MMReqState,
        request: fastapi.Request | None = None,
    ):
        """Wait for results for a single request, yielding responses.

        This method waits on `state.event` with a timeout (`self.wait_timeout`),
        handles client disconnects (aborting the request), and yields
        intermediate/final outputs according to `obj.stream`.
        """
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    ) from None
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    )

    async def create_speech(
        self,
        obj: AudioSpeechRequest,
        request: fastapi.Request | None = None,
    ) -> bytes:
        """OpenAI-compatible TTS: convert text to audio.

        Args:
            obj: AudioSpeechRequest containing text and voice parameters.
            request: FastAPI request object for disconnect handling.

        Returns:
            Raw audio bytes in the specified format.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()
        rid = uuid.uuid4().hex

        # Tokenize the text input
        text_input_ids = None
        if obj.input and self.tokenizer is not None:
            encoded = self.tokenizer(obj.input)
            text_input_ids = encoded["input_ids"]

        # Optional: encode voice/instructions as prompt
        prompt_input_ids = None
        if obj.instructions and self.tokenizer is not None:
            encoded = self.tokenizer(obj.instructions)
            prompt_input_ids = encoded["input_ids"]

        from sgl_jax.srt.multimodal.manager.schedule_batch import Req

        tts_req = Req(
            rid=rid,
            audio_mode="tts",
            text=obj.input,
            text_input_ids=text_input_ids,
            prompt=obj.instructions,
            prompt_input_ids=prompt_input_ids,
            data_type=DataType.AUDIO,
            sample_rate=24000,  # Default sample rate, can be adjusted based on response_format
        )

        state = MMReqState(
            rid=rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[rid] = state

        self.send_to_scheduler.send_pyobj(tts_req)

        try:
            await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
        except TimeoutError:
            raise ValueError(f"TTS request timed out for rid={rid}") from None

        del self.rid_to_state[rid]

        out = state.out_list[-1] if state.out_list else {}

        # Convert audio to specified format
        if out.get("audio_data") is not None:
            audio_array = out["audio_data"]
            # TODO: Convert to obj.response_format (mp3, wav, pcm, etc.)
            # For now, return raw float32 bytes (PCM)
            audio_bytes = audio_array.astype(np.float32).tobytes()
            return audio_bytes
        else:
            raise ValueError("No audio data generated")

    async def create_transcription(
        self,
        obj: AudioTranscriptionRequest,
        request: fastapi.Request | None = None,
    ) -> AudioTranscriptionResponse | str:
        """OpenAI-compatible ASR: convert audio to text.

        Supports both file upload and URL download (handled by HTTP endpoint).

        Args:
            obj: AudioTranscriptionRequest containing audio data and parameters.
            request: FastAPI request object for disconnect handling.

        Returns:
            Transcription in the specified format (AudioTranscriptionResponse or str).
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()
        rid = uuid.uuid4().hex

        # Get audio bytes (already processed by HTTP endpoint)
        if obj.file is None:
            raise ValueError("Audio file is required (should be handled by HTTP endpoint)")

        # Load audio file
        audio_array = self._load_audio_from_bytes(obj.file, target_sr=24000)

        # Preprocess to mel spectrogram
        mel_input, mel_input_lens = self._preprocess_audio_to_mel(audio_array)

        # Build prompt (following MiMo format)
        prefix_ids = None
        suffix_ids = None
        if self.tokenizer is not None:
            prefix_text = "<|im_start|>user\n"
            prompt_text = obj.prompt or "请转录这段音频"
            suffix_text = f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            prefix_ids = self.tokenizer(prefix_text)["input_ids"]
            suffix_ids = self.tokenizer(suffix_text)["input_ids"]

        from sgl_jax.srt.multimodal.manager.schedule_batch import Req

        asr_req = Req(
            rid=rid,
            mel_input=mel_input,
            mel_input_lens=mel_input_lens,
            audio_mode="asr",
            sample_rate=24000,
            data_type=DataType.AUDIO,
            text_input_ids=suffix_ids,
            prompt_input_ids=prefix_ids,
            prompt=obj.prompt,
            n_q=8,
        )

        state = MMReqState(
            rid=rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[rid] = state

        self.send_to_scheduler.send_pyobj(asr_req)

        try:
            await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
        except TimeoutError:
            raise ValueError(f"ASR request timed out for rid={rid}") from None

        del self.rid_to_state[rid]

        out = state.out_list[-1] if state.out_list else {}

        # Decode text
        text = ""
        if out.get("text") is not None:
            text = out["text"]
        elif out.get("generated_text_tokens") is not None and self.tokenizer is not None:
            tokens = out["generated_text_tokens"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        # Return different formats based on response_format
        if obj.response_format == "text":
            return text
        elif obj.response_format == "srt":
            # TODO: Generate SRT subtitle format
            return self._format_as_srt(text, out.get("segments"))
        elif obj.response_format == "vtt":
            # TODO: Generate VTT subtitle format
            return self._format_as_vtt(text, out.get("segments"))
        else:  # json, verbose_json, diarized_json
            return AudioTranscriptionResponse(
                text=text,
                task="transcribe",
                language=obj.language,
                # TODO: Add duration, segments, usage fields
            )

    def _format_as_srt(self, text: str, segments: list[dict] | None) -> str:
        """Format transcription as SRT subtitles.

        TODO: Implement SRT formatting with timestamps.
        """
        # Placeholder implementation
        return f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"

    def _format_as_vtt(self, text: str, segments: list[dict] | None) -> str:
        """Format transcription as WebVTT subtitles.

        TODO: Implement VTT formatting with timestamps.
        """
        # Placeholder implementation
        return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n"

    def _load_audio_from_bytes(self, audio_bytes: bytes, target_sr: int = 24000) -> np.ndarray:
        """Load audio from bytes (wav, mp3, etc.) and resample to target_sr.

        Uses soundfile for loading and torchaudio for resampling to match
        the official MiMo Audio implementation.

        Note: librosa.load produces different resampling results that cause
        ~2.5% of mel spectrogram values to hit the floor, affecting ASR accuracy.
        """
        import soundfile as sf
        import torch
        import torchaudio

        # Check if it's a valid audio file format
        if audio_bytes[:4] == b'RIFF':
            logger.debug("Detected WAV format (RIFF header)")
        elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            logger.debug("Detected MP3 format")
        elif audio_bytes[:4] == b'fLaC':
            logger.debug("Detected FLAC format")
        elif audio_bytes[:4] == b'OggS':
            logger.debug("Detected OGG format")
        else:
            logger.warning("Unknown audio format, first 4 bytes: %s", audio_bytes[:4].hex())

        # Load audio with soundfile (returns numpy array)
        with io.BytesIO(audio_bytes) as f:
            audio_array, orig_sr = sf.read(f)

        # Convert to torch tensor for resampling
        audio_tensor = torch.from_numpy(audio_array).float()

        # Handle stereo -> mono (average channels like official impl)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.mean(dim=1)

        # Resample using torchaudio (matches official MiMo implementation)
        if orig_sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
            logger.debug("Resampled audio from %d Hz to %d Hz", orig_sr, target_sr)

        audio_array = audio_tensor.numpy()
        logger.debug("Audio loaded: orig_sr=%d, target_sr=%d, samples=%d",
                     orig_sr, target_sr, len(audio_array))
        return audio_array

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using torchaudio.

        Uses torchaudio.functional.resample to match official MiMo implementation.

        Args:
            audio: Input audio array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio

        import torch
        import torchaudio

        audio_tensor = torch.from_numpy(audio).float()
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        logger.info("Resampled audio from %d Hz to %d Hz (%d -> %d samples)",
                    orig_sr, target_sr, len(audio), len(resampled))
        return resampled.numpy().astype(np.float32)

    async def chat_completion_audio(
        self,
        obj: GenerateOpenAIAudioInput,
        request: fastapi.Request | None = None,
    ):
        """OpenAI-compatible chat completion for multimodal audio.

        Args:
            obj: GenerateOpenAIAudioInput containing messages and audio config.
            request: FastAPI request object.

        Returns:
            A dict formatted as an OpenAI Chat Completion response.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        rid = uuid.uuid4().hex

        # 1. Parse messages to extract prompt and audio
        prompt_text = ""
        audio_data_bytes = None
        
        for msg in obj.messages:
            if isinstance(msg.content, str):
                if msg.role == "user":
                    prompt_text += msg.content + "\n"
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == "text":
                        prompt_text += part.text + "\n"
                    elif part.type == "input_audio":
                        # We only support one audio input for now (MiMo Audio limitation)
                        if part.input_audio:
                            if part.input_audio.data:
                                try:
                                    audio_data_bytes = base64.b64decode(part.input_audio.data)
                                except Exception:
                                    logger.warning("Failed to decode base64 audio in chat_completion")
                            elif part.input_audio.url:
                                try:
                                    # Download audio from URL
                                    logger.info("Downloading audio from URL: %s", part.input_audio.url)
                                    # Using a timeout to prevent hanging
                                    resp = requests.get(part.input_audio.url, timeout=30)
                                    resp.raise_for_status()
                                    audio_data_bytes = resp.content
                                    logger.info("Downloaded %d bytes", len(audio_data_bytes))
                                except Exception as e:
                                    logger.warning("Failed to download audio from URL: %s", e)

        # 2. Construct Prompt Segments (Manual construction for precise Audio placement)
        # Official MiMo format: <|im_start|>user\n[AUDIO]{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n
        # Audio comes FIRST, then instruction/question

        prefix_text = "<|im_start|>user\n"
        suffix_text = f"{prompt_text.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        
        prefix_ids = []
        suffix_ids = []
        
        if self.tokenizer:
            try:
                # Tokenize prefix
                prefix_ids = self.tokenizer(prefix_text)["input_ids"]
                # Tokenize suffix
                suffix_ids = self.tokenizer(suffix_text)["input_ids"]
            except Exception as e:
                logger.warning("Tokenization error in chat_completion_audio: %s", e)

        # 3. Process audio if present
        mel_input = None
        mel_input_lens = None
        if audio_data_bytes:
            # Use our smart loader (handles WAV/MP3/etc.)
            audio_array = self._load_audio_from_bytes(audio_data_bytes)
            mel_input, mel_input_lens = self._preprocess_audio_to_mel(audio_array)

        # 4. Create internal Req and send to scheduler
        from sgl_jax.srt.multimodal.manager.schedule_batch import Req
        
        # We use audio_understanding mode for general chat with audio
        internal_req = Req(
            rid=rid,
            mel_input=mel_input,
            mel_input_lens=mel_input_lens,
            audio_mode="audio_understanding",
            text_input_ids=suffix_ids,
            prompt_input_ids=prefix_ids,
            data_type=DataType.AUDIO,
            sample_rate=24000,
            prompt=prompt_text,
            n_q=8,
            max_new_tokens=obj.max_tokens or 256,
        )

        state = MMReqState(
            rid=rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[rid] = state

        self.send_to_scheduler.send_pyobj(internal_req)

        # 5. Wait for result
        try:
            await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
        except TimeoutError:
            raise ValueError(f"Chat completion request timed out for rid={rid}") from None

        del self.rid_to_state[rid]
        out = state.out_list[-1] if state.out_list else {"success": True, "meta_info": {}}

        # 6. Format response as OpenAI Chat Completion
        result_text = out.get("text", "")
        
        # Extract raw tokens if text is empty (debug fallback)
        if not result_text and out.get("generated_text_tokens") is not None and self.tokenizer is not None:
            tokens = out["generated_text_tokens"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            result_text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        return {
            "id": f"chatcmpl-{rid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": obj.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prefix_ids) + len(suffix_ids),
                "completion_tokens": len(out.get("generated_text_tokens", [])),
                "total_tokens": len(prefix_ids) + len(suffix_ids) + len(out.get("generated_text_tokens", [])),
            }
        }


def run_multimodal_tokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::multimodal_tokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        tokenizer = MultimodalTokenizer(server_args, port_args)
        tokenizer.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("MultimodalTokenizerManager hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)

    return tokenizer
