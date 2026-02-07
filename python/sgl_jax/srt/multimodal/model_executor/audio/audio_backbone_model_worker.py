"""Audio Backbone Model Worker for MiMo Audio."""

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_runner import (
    AudioBackboneModelRunner,
)
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Group size for MiMo Audio backbone
MIMO_GROUP_SIZE = 4


class AudioBackboneModelWorker:
    """Worker for MiMo Audio Backbone model execution."""

    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioBackboneModelRunner(server_args, mesh, model_class=model_class)

    def _create_forward_batch(
        self,
        input_ids: jax.Array,
        positions_override: jax.Array = None,
        is_prefill: bool = True,
    ) -> ForwardBatch:
        """Create ForwardBatch for RadixAttention.

        Args:
            input_ids: [B, channels, seq_len]
            positions_override: Optional positions array to use instead of computing
            is_prefill: Whether this is prefill or decode phase

        Returns:
            ForwardBatch with positions and metadata for RadixAttention
        """
        B, _, seq_len = input_ids.shape
        T_groups = seq_len // MIMO_GROUP_SIZE  # Number of groups (positions for main transformer)

        # Use provided positions or compute based on mode
        if positions_override is not None:
            positions = positions_override
            # Determine mode based on positions length
            if positions.shape[0] == T_groups:
                forward_mode = ForwardMode.EXTEND
                extend_start_loc = jnp.array([0], dtype=jnp.int32)
                extend_seq_lens = jnp.array([T_groups] * B, dtype=jnp.int32)
                extend_prefix_lens = jnp.zeros(B, dtype=jnp.int32)
            else:
                forward_mode = ForwardMode.DECODE
                extend_start_loc = None
                extend_seq_lens = None
                extend_prefix_lens = None
        elif is_prefill:
            # Prefill: positions for all groups
            positions = jnp.arange(T_groups, dtype=jnp.int32)
            forward_mode = ForwardMode.EXTEND
            extend_start_loc = jnp.array([0], dtype=jnp.int32)
            extend_seq_lens = jnp.array([T_groups] * B, dtype=jnp.int32)
            extend_prefix_lens = jnp.zeros(B, dtype=jnp.int32)
        else:
            # Decode: single position
            positions = jnp.array([T_groups - 1], dtype=jnp.int32)
            forward_mode = ForwardMode.DECODE
            extend_start_loc = None
            extend_seq_lens = None
            extend_prefix_lens = None

        # Create req_pool_indices - allocate from req_to_token_pool
        req_pool_indices = np.arange(B, dtype=np.int32)

        # Create seq_lens based on T_groups
        seq_lens_arr = np.array([T_groups] * B, dtype=np.int32)

        # Allocate KV cache slots
        total_tokens = T_groups * B
        out_cache_loc = self.model_runner.token_to_kv_pool_allocator.alloc(total_tokens)
        if out_cache_loc is None:
            raise RuntimeError("Failed to allocate KV cache slots")

        return ForwardBatch(
            bid=0,  # Batch id
            forward_mode=forward_mode,
            batch_size=B,
            input_ids=None,  # We pass input_ids separately to model
            positions=positions,
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array(seq_lens_arr),
            out_cache_loc=jnp.array(out_cache_loc),
            cache_loc=jnp.array(out_cache_loc),  # Same as out_cache_loc for simple case
            extend_start_loc=extend_start_loc,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            attn_backend=self.model_runner.attn_backend,
        )

    def _create_logits_metadata(
        self,
        batch_size: int,
        seq_len: int,
        is_prefill: bool = True,
    ) -> LogitsMetadata:
        """Create LogitsMetadata for logits processing.

        Args:
            batch_size: Number of requests in batch
            seq_len: Sequence length (T_groups for MiMo)
            is_prefill: Whether this is prefill or decode phase

        Returns:
            LogitsMetadata for LogitsProcessor
        """
        if is_prefill:
            return LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                extend_seq_lens=jnp.array([seq_len] * batch_size, dtype=jnp.int32),
                extend_seq_lens_cpu=[seq_len] * batch_size,
                top_logprobs_nums=[0] * batch_size,
            )
        else:
            return LogitsMetadata(
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                top_logprobs_nums=[0] * batch_size,
            )

    def forward(
        self,
        batch: Req,
        cache: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass through main transformer.

        The input_ids should already be in the correct format [B, 9, seq_len]
        after aggregation in Req._build_backbone_input().

        Args:
            batch: Request batch containing pre-aggregated input_ids
            cache: Optional KV cache (unused, for interface compatibility)
            positions: Optional positions array from kwargs

        Returns:
            (text_logits, local_hidden_states, cache), cache_miss_count
        """
        input_ids = batch.input_ids

        if input_ids is None:
            raise ValueError("input_ids must be provided (should be pre-aggregated by Req)")

        logger.info(
            "AudioBackboneModelWorker.forward: input_ids shape=%s, dtype=%s",
            input_ids.shape if input_ids is not None else None,
            input_ids.dtype if input_ids is not None else None,
        )

        # Ensure correct dtype
        if not jnp.issubdtype(input_ids.dtype, jnp.integer):
            input_ids = input_ids.astype(jnp.int32)

        # Ensure batch dimension
        if input_ids.ndim == 2:
            # [9, seq_len] -> [1, 9, seq_len]
            input_ids = input_ids[None, :, :]

        # Ensure seq_len is divisible by group_size
        seq_len = input_ids.shape[2]
        if seq_len % MIMO_GROUP_SIZE != 0:
            pad_len = MIMO_GROUP_SIZE - (seq_len % MIMO_GROUP_SIZE)
            input_ids = jnp.pad(
                input_ids,
                ((0, 0), (0, 0), (0, pad_len)),
                constant_values=0,
            )
            logger.info(
                "Padded input_ids from seq_len=%d to %d",
                seq_len,
                input_ids.shape[2],
            )

        B, _, padded_seq_len = input_ids.shape
        T_groups = padded_seq_len // MIMO_GROUP_SIZE

        # Get positions from kwargs if provided
        positions_override = kwargs.get("positions", None)

        # Determine prefill mode: prefill if no positions provided
        is_prefill = positions_override is None

        # Create ForwardBatch and LogitsMetadata
        forward_batch = self._create_forward_batch(input_ids, positions_override, is_prefill)
        logits_metadata = self._create_logits_metadata(B, T_groups, forward_batch.forward_mode.is_extend())

        return self.model_runner.forward(input_ids, forward_batch, logits_metadata, **kwargs)

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        """Generate audio tokens for one group.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            sampler_config: Sampling configuration

        Returns:
            local_tokens: [B, group_size, audio_channels], cache_miss_count
        """
        return self.model_runner.patch_decode(local_embeds, key, sampler_config)
