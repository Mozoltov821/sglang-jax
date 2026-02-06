"""Audio Backbone Model Runner for MiMo Audio."""

import json
import logging
import os
from functools import partial
from typing import Optional

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_loader.loader import get_model_loader
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import (
    MiMoAudioArguments,
    MiMoAudioBackboneConfig,
    MiMoSamplerConfig,
)
from sgl_jax.srt.multimodal.configs.config_registry import get_audio_backbone_config
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
class AudioBackboneModelRunner(BaseModelRunner):
    """Model runner for MiMo Audio Backbone (LLM with audio generation)."""

    def __init__(
        self,
        server_args: ServerArgs = None,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
    ):
        self.mesh = mesh
        self.model_loader = get_model_loader(
            load_config=LoadConfig(
                model_class=model_class,
                sub_dir=None,
            ),
            mesh=self.mesh,
        )
        self.model_class = model_class
        self.server_args = server_args
        self.initialize()

    def initialize(self):
        self.load_model()
        self.page_size = 1  # Set before init_attention_backend which uses it
        self.init_attention_backend()
        self.init_memory_pool()
        self.initialize_jit()

    def init_attention_backend(self):
        """Init attention kernel backend."""
        self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self):
        # Force NativeAttention for MiMo Audio backbone to avoid shard_map mesh context issues
        # FlashAttention uses jax.shard_map which requires mesh context during JIT tracing
        return NativeAttention(
            self.model_config.num_attention_heads,
            self.model_config.num_key_value_heads,
            self.mesh,
        )

    def init_memory_pool(self):
        """Initialize memory pool for KV cache."""
        # Simple fixed size pool for now
        self.max_total_num_tokens = 32768
        self.page_size = 1 # Simple indexing
        
        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=jnp.bfloat16,
            head_num=self.model_config.num_key_value_heads, # Assume TP=1 for now
            head_dim=self.model_config.head_dim,
            layer_num=self.model_config.num_hidden_layers,
            mesh=self.mesh,
        )
        logger.info("Initialized KV Memory Pool with size %d", self.max_total_num_tokens)

    def _load_hf_config(self, model_path: str) -> dict:
        """Load config.json from HuggingFace model path."""
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, "config.json")
        else:
            config_path = huggingface_hub.hf_hub_download(
                model_path,
                "config.json",
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        with open(config_path, "r") as f:
            return json.load(f)

    def load_model(self):
        hf_config = self._load_hf_config(self.server_args.model_path)
        self.model_config = get_audio_backbone_config(self.server_args.model_path)

        # Update config with values from HF config
        for key, value in hf_config.items():
            if hasattr(self.model_config, key):
                if key in ("speech_vocab_sizes", "speech_empty_ids"):
                    if isinstance(value, str):
                        import ast
                        value = tuple(ast.literal_eval(value))
                    elif isinstance(value, list):
                        value = tuple(value)
                elif key == "delay_pattern":
                    if isinstance(value, str) and "-" in value:
                        value = tuple(int(x) for x in value.split("-"))
                    elif isinstance(value, str):
                        import ast
                        value = tuple(ast.literal_eval(value))
                    elif isinstance(value, list):
                        value = tuple(value)
                setattr(self.model_config, key, value)

        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class

        # Create audio arguments from HF config
        self.audio_args = MiMoAudioArguments(
            model_name_or_path=self.server_args.model_path,
            sosp_idx=hf_config.get("sosp_idx", 0),
            eosp_idx=hf_config.get("eosp_idx", 0),
            sostm_idx=hf_config.get("sostm_idx", 0),
            eostm_idx=hf_config.get("eostm_idx", 0),
            eot_idx=hf_config.get("eot_idx", 0),
            empty_idx=hf_config.get("empty_idx", 0),
        )

        self.model = self.model_loader.load_model(
            model_config=self.model_config,
        )

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        # Define JIT functions within mesh context so shard_map can access mesh
        with self.mesh:
            @partial(
                jax.jit,
                donate_argnames=["token_to_kv_pool"],
                static_argnames=["model_state_def"],
            )
            def forward(
                model_def,
                model_state_def,
                model_state_leaves,
                input_ids,
                forward_batch,
                token_to_kv_pool,
                logits_metadata,
            ):
                model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
                model = nnx.merge(model_def, model_state)
                return model.forward(input_ids, forward_batch, token_to_kv_pool, logits_metadata)

            @partial(
                jax.jit,
                static_argnames=["model_state_def", "do_sample", "temperature"],
            )
            def patch_decode(
                model_def,
                model_state_def,
                model_state_leaves,
                local_embeds,
                key,
                do_sample,
                temperature,
            ):
                model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
                model = nnx.merge(model_def, model_state)
                sampler_config = MiMoSamplerConfig(do_sample=do_sample, temperature=temperature)
                return model.patch_decode(local_embeds, key, sampler_config)

        def forward_wrapper(
            input_ids: jax.Array,
            forward_batch: ForwardBatch,
            logits_metadata: LogitsMetadata,
        ):
            return forward(
                model_def,
                model_state_def,
                model_state_leaves,
                input_ids,
                forward_batch,
                self.token_to_kv_pool,
                logits_metadata
            )

        def patch_decode_wrapper(
            local_embeds: jax.Array,
            key: jax.Array,
            sampler_config: Optional[MiMoSamplerConfig] = None,
        ):
            if sampler_config is None:
                sampler_config = MiMoSamplerConfig()
            return patch_decode(
                model_def,
                model_state_def,
                model_state_leaves,
                local_embeds,
                key,
                sampler_config.do_sample,
                sampler_config.temperature,
            )

        self.jitted_forward = forward_wrapper
        self.jitted_patch_decode = patch_decode_wrapper

    def forward(
        self,
        input_ids: jax.Array,
        cache: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass through main transformer using RadixAttention.

        Args:
            input_ids: [B, 1 + audio_channels, seq_len]
            cache: Unused (legacy)
            kwargs: Must contain 'positions'

        Returns:
            (text_logits, local_hidden_states, None), cache_miss_count
        """
        cache_miss_count = 0
        positions = kwargs.get("positions", None)
        
        # Infer positions if not provided (Prefill)
        if positions is None:
            B, _, L = input_ids.shape
            positions = jnp.arange(L, dtype=jnp.int32)
        
        # Determine Forward Mode
        is_prefill = (positions[0] == 0)
        forward_mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
        
        # Simple Request Pool Indexing: Identity map position -> pool index
        req_pool_indices = positions
        out_cache_loc = positions
        
        # Seq Lens: [B]
        seq_lens = jnp.array([positions[-1] + 1], dtype=jnp.int32)
        
        # Construct ForwardBatch
        forward_batch = ForwardBatch(
            bid=0, # Dummy bid
            forward_mode=forward_mode,
            batch_size=1,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            positions=positions,
            extend_start_loc=jnp.array([0], dtype=jnp.int32) if is_prefill else None,
            attn_backend=self.attn_backend,
        )

        if self.token_to_kv_pool is None:
            raise ValueError("Token pool is None")

        # Construct LogitsMetadata
        logits_metadata = LogitsMetadata(
            forward_mode=forward_mode,
            # For MiMo, we only care about the last token's logits for ASR
            # In decode mode, total_num_tokens is 1. In prefill, it is L.
            # LogitsProcessor usually needs to know which positions to extract.
            # Qwen2 style expects extract_last_only logic.
        )

        import jax._src.test_util as jtu

        with self.mesh:
            with jtu.count_pjit_cpp_cache_miss() as count:
                # output is (text_logits, local_hidden_states, None, layers_kv_fused, layers_callback_flag)
                text_logits, local_hidden_states, _, layers_kv_fused, _ = self.jitted_forward(
                    input_ids, forward_batch, logits_metadata
                )
                cache_miss_count = count()

            # Update KV Cache Buffer in Pool
            self.token_to_kv_pool.replace_kv_buffer(layers_kv_fused)
        
        return (text_logits, local_hidden_states, None), cache_miss_count

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        """Generate audio tokens for one group using patch decoder.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            sampler_config: Sampling configuration

        Returns:
            local_tokens: [B, group_size, audio_channels], cache_miss_count
        """
        cache_miss_count = 0
        import jax._src.test_util as jtu

        with self.mesh:
            with jtu.count_pjit_cpp_cache_miss() as count:
                output = self.jitted_patch_decode(local_embeds, key, sampler_config)
                cache_miss_count = count()
        return output, cache_miss_count
