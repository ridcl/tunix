# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5 model.

Qwen3.5 is a hybrid architecture combining full (causal) attention layers with
linear attention layers (GatedDeltaNet).  It also integrates a vision encoder
identical to Qwen3-VL but without the deepstack feature.

New vs Qwen3-VL:
  - Hybrid decoder: alternating ``linear_attention`` / ``full_attention`` layers.
  - ``GatedDeltaNet``: linear-time recurrent layer based on the gated delta
    rule (https://arxiv.org/abs/2406.06484).
  - Gated query in full-attention: ``q_proj`` produces query + a sigmoid gate
    that is applied element-wise to the attention output.
  - Partial RoPE: only the first ``head_dim * partial_rotary_factor`` dims of
    each head are rotated; the rest pass through unchanged.
  - ``RMSNorm`` uses a *residual* weight ``(1 + w)`` initialised at zero so
    that the network starts as an identity (Qwen3-Next style).
"""

import dataclasses
from typing import Tuple

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
# Re-use pure utilities from Qwen3-VL that do not depend on its ModelConfig.
from tunix.models.qwen3vl.model import Cache
from tunix.models.qwen3vl.model import Einsum
from tunix.models.qwen3vl.model import Embedder
from tunix.models.qwen3vl.model import K_MASK
from tunix.models.qwen3vl.model import LayerCache
from tunix.models.qwen3vl.model import make_causal_mask_from_positions
from tunix.models.qwen3vl.model import RematConfig
from tunix.models.qwen3vl.model import shard
from tunix.models.qwen3vl.model import ShardingConfig
from tunix.models.qwen3vl.vision import VisionEmbeddings
from tunix.models.qwen3vl.vision import VisionGridData
from tunix.models.qwen3vl.vision import VisionModel
from tunix.models.qwen3vl.vision import VisionModelConfig
from tunix.utils import compat
from tunix.utils import env_utils

env_utils.setup_sharding_environment()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def qwix_friendly_scan(fn, init, xs, **kwargs):
  """Like jax.lax.scan, but safe to call inside a QWIX-intercepted model.

  QWIX sets ``jax_disable_jit=True`` globally while intercepting a model's
  forward pass so that its ``dot_general`` code-object patch takes effect.
  With JIT disabled, ``jax.lax.scan`` executes the body eagerly once per
  element instead of tracing it once, which causes an O(seq_len) hang for
  models that use scan in their decoder layers.  This wrapper temporarily
  re-enables JIT around the scan call so that it always produces a compact
  traced loop.
  """
  jit_was_disabled = jax.config.jax_disable_jit
  if jit_was_disabled:
    jax.config.update("jax_disable_jit", False)
  try:
    return jax.lax.scan(fn, init, xs, **kwargs)
  finally:
    if jit_was_disabled:
      jax.config.update("jax_disable_jit", True)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for the Qwen3.5 model."""

  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  # Hybrid layer types: list of "full_attention" | "linear_attention".
  # Default pattern: every 4th layer is full attention.
  layer_types: list[str]
  # Partial RoPE: fraction of each head dimension that receives rotation.
  partial_rotary_factor: float = 0.25
  # M-RoPE section sizes for the text model (temporal, height, width).
  mrope_section: Tuple[int, ...] = (11, 11, 10)
  use_tied_embedding: bool = False
  # Linear attention (GatedDeltaNet) parameters.
  linear_conv_kernel_dim: int = 4
  linear_key_head_dim: int = 128
  linear_value_head_dim: int = 128
  linear_num_key_heads: int = 16
  linear_num_value_heads: int = 32
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  param_dtype: jnp.dtype = jnp.bfloat16
  vision_config: VisionModelConfig | None = None

  @classmethod
  def qwen3_5_0p8b(cls):
    return cls(
        num_layers=24,
        vocab_size=248320,
        embed_dim=1024,
        hidden_dim=3584,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        use_tied_embedding=True,
        layer_types=[
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(24)
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
    )

  @classmethod
  def qwen3_5_4b(cls):
    return cls(
        num_layers=32,
        vocab_size=248320,
        embed_dim=2560,
        hidden_dim=9216,
        num_heads=16,
        head_dim=256,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        use_tied_embedding=True,
        layer_types=[
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(32)
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
    )

  @classmethod
  def qwen3_5_9b(cls):
    return cls(
        num_layers=32,
        vocab_size=248320,
        embed_dim=4096,
        hidden_dim=12288,
        num_heads=16,
        head_dim=256,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        partial_rotary_factor=0.25,
        mrope_section=(11, 11, 10),
        layer_types=[
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(32)
        ],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
    )

  # TODO: Does it even make sense to load models without vision encoder?
  # TODO: Does 0.8B model support vision encoder?
  @classmethod
  def qwen3_5_0p8b_vl(cls):
    """Qwen3.5-0.8B with vision encoder."""
    cfg = cls.qwen3_5_0p8b()
    cfg.vision_config = VisionModelConfig(
        hidden_size=768,
        out_hidden_size=1024,
        depth=12,
        num_heads=12,
        intermediate_size=3072,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=0,
        in_channels=3,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(),
        mrope_section=(11, 11, 10),
        image_pad_id=248056,
    )
    return cfg

  @classmethod
  def qwen3_5_4b_vl(cls):
    """Qwen3.5-4B with vision encoder."""
    cfg = cls.qwen3_5_4b()
    cfg.vision_config = VisionModelConfig(
        hidden_size=1024,
        out_hidden_size=2560,
        depth=24,
        num_heads=16,
        intermediate_size=4096,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=0,
        in_channels=3,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(),
        mrope_section=(11, 11, 10),
        image_pad_id=248056,
    )
    return cfg

  @classmethod
  def qwen3_5_9b_vl(cls):
    """Qwen3.5-9B with vision encoder."""
    cfg = cls.qwen3_5_9b()
    cfg.vision_config = VisionModelConfig(
        hidden_size=1152,
        out_hidden_size=3584,
        depth=27,
        num_heads=16,
        intermediate_size=4304,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=0,
        in_channels=3,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(),
        mrope_section=(11, 11, 10),
        image_pad_id=248056,
    )
    return cfg


# ---------------------------------------------------------------------------
# RMSNorm variants
# ---------------------------------------------------------------------------


class RMSNorm(nnx.Module):
  """RMSNorm with residual weight ``(1 + w)`` (Qwen3-Next / Qwen3.5 style).

  Weight is initialised to *zero* so the initial mapping is the identity.
  """

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.w = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), dim).astype(param_dtype),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps

  @jax.named_scope("rms_norm")
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    rms_inv = jax.lax.rsqrt(
        jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.norm_eps
    )
    normed = (x_f32 * rms_inv).astype(dtype)
    return normed * (1.0 + self.w)


class RMSNormGated(nnx.Module):
  """RMS norm with a SiLU gate; used inside GatedDeltaNet.

  ``forward(x, gate)`` computes ``rms_norm(x) * silu(gate)`` where both
  ``x`` and ``gate`` have shape ``[*, head_v_dim]``.
  """

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),
    )
    self.norm_eps = norm_eps

  @jax.named_scope("rms_norm_gated")
  def __call__(
      self, x: jaxtyping.Array, gate: jaxtyping.Array
  ) -> jaxtyping.Array:
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    rms_inv = jax.lax.rsqrt(
        jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.norm_eps
    )
    normed = self.w * (x_f32 * rms_inv).astype(dtype)
    return normed * jax.nn.silu(gate.astype(jnp.float32)).astype(dtype)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [3, B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
    mrope_section: tuple[int, ...] = (11, 11, 10),
    partial_rotary_factor: float = 0.25,
) -> jaxtyping.Array:
  """M-RoPE with optional partial rotation.

  Only the first ``rotary_dim = head_dim * partial_rotary_factor`` dimensions
  of each head are rotated; the remaining dimensions pass through unchanged.
  """
  rotary_dim = int(head_dim * partial_rotary_factor)
  freq_dim = rotary_dim // 2  # number of (cos, sin) pairs

  fraction = 2 * jnp.arange(0, freq_dim, dtype=jnp.float32) / rotary_dim
  timescale = rope_theta**fraction  # [freq_dim]

  # [3, B, L, freq_dim]
  sinusoid_inp = (
      positions[:, :, :, jnp.newaxis].astype(jnp.float32)
      / timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
  )

  # Interleaved M-RoPE: T axis is base; H and W overwrite their slots.
  freq = sinusoid_inp[0]  # [B, L, freq_dim] — T axis
  h_idx = jnp.arange(1, mrope_section[1] * 3, 3)
  w_idx = jnp.arange(2, mrope_section[2] * 3, 3)
  freq = freq.at[..., h_idx].set(sinusoid_inp[1][..., h_idx])
  freq = freq.at[..., w_idx].set(sinusoid_inp[2][..., w_idx])

  freq = freq[:, :, jnp.newaxis, :]  # [B, L, 1, freq_dim]
  sin = jnp.sin(freq)
  cos = jnp.cos(freq)

  if rotary_dim < head_dim:
    to_rotate = inputs[..., :rotary_dim]  # [B, L, N, rotary_dim]
    passthrough = inputs[..., rotary_dim:]  # [B, L, N, head_dim - rotary_dim]
    first_half, second_half = jnp.split(to_rotate, 2, axis=-1)
    rotated = jnp.concatenate(
        [
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ],
        axis=-1,
    )
    out = jnp.concatenate([rotated, passthrough], axis=-1)
  else:
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    out = jnp.concatenate(
        [
            first_half * cos - second_half * sin,
            second_half * cos + first_half * sin,
        ],
        axis=-1,
    )

  return out.astype(inputs.dtype)


# ---------------------------------------------------------------------------
# GatedDeltaNet helpers
# ---------------------------------------------------------------------------


def _l2norm(x: jaxtyping.Array, eps: float = 1e-6) -> jaxtyping.Array:
  """L2-normalise along the last axis."""
  inv_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
  return x * inv_norm


def _causal_conv1d_fwd(
    x: jaxtyping.Array,  # [B, T, C]
    weight: jaxtyping.Array,  # [kernel_size, C]  depthwise
) -> jaxtyping.Array:  # [B, T, C]
  """Causal depthwise conv1d followed by SiLU (prefill path)."""
  kernel_size = weight.shape[0]
  # Left-pad so output[t] only sees input[t-K+1 .. t].
  pad = jnp.zeros((x.shape[0], kernel_size - 1, x.shape[2]), dtype=x.dtype)
  x_pad = jnp.concatenate([pad, x], axis=1)  # [B, T+K-1, C]
  # Stack kernel-sized windows along a new axis.
  windows = jnp.stack(
      [x_pad[:, k : k + x.shape[1], :] for k in range(kernel_size)], axis=2
  )  # [B, T, K, C]
  out = jnp.sum(windows * weight[jnp.newaxis, jnp.newaxis, :, :], axis=2)
  return jax.nn.silu(out)  # [B, T, C]


def _causal_conv1d_update(
    x_new: jaxtyping.Array,  # [B, 1, C]
    conv_state: jaxtyping.Array,  # [B, K-1, C]  rolling buffer
    weight: jaxtyping.Array,  # [K, C]  depthwise
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  """Single-step causal conv1d update (decode path).

  Returns ``(output [B, 1, C], new_conv_state [B, K-1, C])``.
  """
  # Assemble full window: [B, K, C]
  x_window = jnp.concatenate([conv_state, x_new], axis=1)
  out = jnp.sum(x_window * weight[jnp.newaxis, :, :], axis=1, keepdims=True)
  out = jax.nn.silu(out)
  # New rolling buffer: drop oldest, keep newest K-1 tokens.
  new_conv_state = x_window[:, 1:, :]
  return out, new_conv_state


def _gated_delta_rule(
    query: jaxtyping.Array,  # [B, T, H, Dk]  — after optional repeat
    key: jaxtyping.Array,  # [B, T, H, Dk]
    value: jaxtyping.Array,  # [B, T, H, Dv]
    g: jaxtyping.Array,  # [B, T, H]  log-decay (negative)
    beta: jaxtyping.Array,  # [B, T, H]
    initial_state: jaxtyping.Array,  # [B, H, Dk, Dv]
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  """Gated delta rule via ``jax.lax.scan`` (works for prefill and decode).

  Returns ``(output [B, T, H, Dv], final_state [B, H, Dk, Dv])``.
  """
  query = _l2norm(query)
  key = _l2norm(key)
  scale = query.shape[-1] ** -0.5
  query = query * scale

  carry_dtype = initial_state.dtype

  def step(
      state: jaxtyping.Array,  # [B, H, Dk, Dv]
      inputs: tuple,
  ) -> tuple[jaxtyping.Array, jaxtyping.Array]:
    q_t, k_t, v_t, g_t, b_t = inputs
    # Compute in float32 for numerical stability, then cast back.
    state_f = state.astype(jnp.float32)
    decay = jnp.exp(g_t)[:, :, jnp.newaxis, jnp.newaxis]  # [B, H, 1, 1]
    state_f = state_f * decay
    kv_mem = jnp.einsum("bhd,bhde->bhe", k_t.astype(jnp.float32), state_f)
    delta = (v_t.astype(jnp.float32) - kv_mem) * b_t[:, :, jnp.newaxis]
    state_f = state_f + jnp.einsum(
        "bhd,bhe->bhde", k_t.astype(jnp.float32), delta
    )
    out_t = jnp.einsum("bhd,bhde->bhe", q_t.astype(jnp.float32), state_f)
    return state_f.astype(carry_dtype), out_t.astype(carry_dtype)

  # Transpose time to leading axis for scan: [T, B, H, *].
  inputs = (
      jnp.moveaxis(query, 1, 0),
      jnp.moveaxis(key, 1, 0),
      jnp.moveaxis(value, 1, 0),
      jnp.moveaxis(g, 1, 0),
      jnp.moveaxis(beta, 1, 0),
  )
  final_state, outputs = qwix_friendly_scan(step, initial_state, inputs)
  # outputs: [T, B, H, Dv] -> [B, T, H, Dv]
  return jnp.moveaxis(outputs, 0, 1), final_state


# ---------------------------------------------------------------------------
# Attention (full attention with gated query — Qwen3-Next / Qwen3.5 style)
# ---------------------------------------------------------------------------


class Attention(nnx.Module):
  """Multi-head attention with a gated query projection.

  The query projection is ``2 x num_heads x head_dim`` wide.  After splitting,
  the second half forms a per-token sigmoid gate that scales the attention
  output before the output projection.
  """

  def __init__(
      self,
      config: "ModelConfig",
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.config = config
    self.shd_config = config.shd_config
    # q_proj is 2× head_dim: first half -> query, second half -> gate.
    self.q_proj = Einsum(
        einsum_str="BTD,DNH->BTNH",
        shape=(config.embed_dim, config.num_heads, config.head_dim * 2),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        param_dtype=param_dtype,
    )
    self.k_proj = Einsum(
        einsum_str="BSD,DKH->BSKH",
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        param_dtype=param_dtype,
    )
    self.v_proj = Einsum(
        einsum_str="BSD,DKH->BSKH",
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        param_dtype=param_dtype,
    )
    self.o_proj = Einsum(
        einsum_str="BTNH,NHD->BTD",
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        param_dtype=param_dtype,
    )
    self.q_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        param_dtype=param_dtype,
    )
    self.k_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        param_dtype=param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = config.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]
    config = self.config

    # q_proj outputs [B, T, num_heads, head_dim*2]; split into query + gate.
    qg = self.q_proj(x)  # [B, T, num_heads, head_dim*2]
    query_proj, gate = jnp.split(
        qg, 2, axis=-1
    )  # each [B, T, num_heads, head_dim]

    key_proj = self.k_proj(x)  # [B, T, kv_heads, head_dim]
    value_proj = self.v_proj(x)  # [B, T, kv_heads, head_dim]

    query_proj = self.q_norm(query_proj)
    key_proj = self.k_norm(key_proj)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = apply_rope(
        query_proj,
        positions,
        head_dim=config.head_dim,
        rope_theta=config.rope_theta,
        mrope_section=config.mrope_section,
        partial_rotary_factor=config.partial_rotary_factor,
    )
    key_proj = apply_rope(
        key_proj,
        positions,
        head_dim=config.head_dim,
        rope_theta=config.rope_theta,
        mrope_section=config.mrope_section,
        partial_rotary_factor=config.partial_rotary_factor,
    )

    if cache is not None:
      end_index = cache["end_index"][0]
      slice_indices = (0, end_index % cache["v"].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache["v"], value_proj, slice_indices
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache["k"], key_proj, slice_indices
      )

    b, t, qh, d = query_proj.shape
    _, s, kh, _ = key_proj.shape

    # GQA
    query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
    attn = jnp.einsum("BTHGD,BSHD->BHGTS", query_proj, key_proj) * self.scale
    attn = attn.reshape((b, qh, t, s))

    if attn_mask is not None:
      attn = jnp.where(jnp.expand_dims(attn_mask, -3), attn, K_MASK)

    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    attn = attn.reshape((b, kh, qh // kh, t, s))
    qkv = jnp.einsum("BHGTS,BSHD->BTHGD", attn, value_proj)
    qkv = qkv.reshape((b, t, qh, d))

    # Apply sigmoid gate before o_proj (gate has same shape as qkv).
    qkv = qkv * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(qkv.dtype)

    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)

    if cache is not None:
      new_cache: LayerCache | None = {
          "v": value_proj,
          "k": key_proj,
          "end_index": cache["end_index"] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, outputs

  @jax.named_scope("attention")
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if self.config.remat_config == RematConfig.BLOCK:
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    return self.block(x, segment_pos, cache, attn_mask)

  @property
  def head_dim(self):
    return self.o_proj.shape[1]


# ---------------------------------------------------------------------------
# GatedDeltaNet (linear attention layer)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nnx.Module):
  """Gated delta-rule linear attention (Qwen3.5 / Qwen3-Next variant).

  Reference: https://arxiv.org/abs/2406.06484

  Differences from the Qwen3-Next variant:
    - Separate ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``
      projections (no combined ``in_proj_qkvz`` / ``in_proj_ba``).
  """

  def __init__(
      self,
      config: "ModelConfig",
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.config = config
    self.num_k_heads = config.linear_num_key_heads
    self.num_v_heads = config.linear_num_value_heads
    self.head_k_dim = config.linear_key_head_dim
    self.head_v_dim = config.linear_value_head_dim
    self.key_dim = self.num_k_heads * self.head_k_dim
    self.value_dim = self.num_v_heads * self.head_v_dim
    self.conv_kernel_size = config.linear_conv_kernel_dim
    # conv_dim covers q, k (key_dim each) and v (value_dim).
    self.conv_dim = self.key_dim * 2 + self.value_dim

    # Input projections.
    self.in_proj_qkv = nnx.Linear(
        config.embed_dim,
        self.conv_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.in_proj_z = nnx.Linear(
        config.embed_dim,
        self.value_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.in_proj_b = nnx.Linear(
        config.embed_dim,
        self.num_v_heads,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.in_proj_a = nnx.Linear(
        config.embed_dim,
        self.num_v_heads,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    # Causal depthwise conv1d weight: [kernel_size, conv_dim].
    self.conv1d_weight = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (self.conv_kernel_size, self.conv_dim)
        )
    )
    # Gated delta-rule parameters.
    self.dt_bias = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), (self.num_v_heads,)).astype(
            param_dtype
        )
    )
    # A_log initialised uniformly in [log(0), log(16)]; we keep it as log(A).
    self.A_log = nnx.Param(
        jnp.log(
            jax.random.uniform(
                rngs.params(), (self.num_v_heads,), minval=0.0, maxval=16.0
            )
        ).astype(param_dtype)
    )
    self.norm = RMSNormGated(
        self.head_v_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.out_proj = nnx.Linear(
        self.value_dim,
        config.embed_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
    )

  @jax.named_scope("gated_delta_net")
  def __call__(
      self,
      x: jaxtyping.Array,  # [B, T, D]
      cache: LayerCache | None,
      attention_mask: jaxtyping.Array | None = None,  # [B, T] padding mask
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    B, T, _ = x.shape

    # Zero out padding positions (mirrors apply_mask_to_padding_states).
    if attention_mask is not None and T > 1:
      x = x * attention_mask[:, :, jnp.newaxis].astype(x.dtype)

    mixed_qkv = self.in_proj_qkv(x)  # [B, T, conv_dim]
    z = self.in_proj_z(x)  # [B, T, value_dim]
    b_raw = self.in_proj_b(x)  # [B, T, num_v_heads]
    a_raw = self.in_proj_a(x)  # [B, T, num_v_heads]

    weight = self.conv1d_weight.value  # [K, conv_dim]

    if cache is not None and T == 1:
      # ── Decode (single-step) path ──────────────────────────────────────────
      conv_state = cache["conv_state"]  # [B, K-1, conv_dim]
      recurrent_state = cache["recurrent_state"]  # [B, H, Dk, Dv]
      mixed_qkv, new_conv_state = _causal_conv1d_update(
          mixed_qkv, conv_state, weight
      )
      new_cache: LayerCache | None = {
          "conv_state": new_conv_state,
          "recurrent_state": None,  # filled below
      }
    else:
      # ── Prefill path ──────────────────────────────────────────────────────
      mixed_qkv = _causal_conv1d_fwd(mixed_qkv, weight)  # [B, T, conv_dim]
      recurrent_state = jnp.zeros(
          (B, self.num_v_heads, self.head_k_dim, self.head_v_dim),
          dtype=x.dtype,
      )
      if cache is not None:
        new_conv_state = mixed_qkv[:, -(self.conv_kernel_size - 1) :, :]
        new_cache = {
            "conv_state": new_conv_state,
            "recurrent_state": None,  # filled below
        }
      else:
        new_cache = None

    # Split into query, key, value.
    query, key, value = jnp.split(
        mixed_qkv, [self.key_dim, self.key_dim * 2], axis=-1
    )
    query = query.reshape(B, T, self.num_k_heads, self.head_k_dim)
    key = key.reshape(B, T, self.num_k_heads, self.head_k_dim)
    value = value.reshape(B, T, self.num_v_heads, self.head_v_dim)

    # Expand key heads to match value heads if needed (GQA-like repeat).
    if self.num_v_heads // self.num_k_heads > 1:
      rep = self.num_v_heads // self.num_k_heads
      query = jnp.repeat(query, rep, axis=2)
      key = jnp.repeat(key, rep, axis=2)

    beta = jax.nn.sigmoid(b_raw)  # [B, T, num_v_heads]
    # g = -exp(A_log) * softplus(a + dt_bias)  (negative log-decay per head)
    g = -(
        jnp.exp(self.A_log.value.astype(jnp.float32))
        * jax.nn.softplus((a_raw + self.dt_bias.value).astype(jnp.float32))
    )  # [B, T, num_v_heads], float32

    core_out, final_state = _gated_delta_rule(
        query, key, value, g, beta, recurrent_state
    )  # core_out: [B, T, num_v_heads, head_v_dim]

    # Update recurrent state in cache.
    if new_cache is not None:
      new_cache["recurrent_state"] = final_state

    # Gated norm.
    z_heads = z.reshape(B, T, self.num_v_heads, self.head_v_dim)
    core_out = self.norm(
        core_out.reshape(-1, self.head_v_dim),
        z_heads.reshape(-1, self.head_v_dim),
    ).reshape(B, T, self.value_dim)

    output = self.out_proj(core_out)  # [B, T, embed_dim]
    return new_cache, output


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nnx.Module):
  """SwiGLU feed-forward network."""

  def __init__(
      self,
      config: "ModelConfig",
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.shd_config = config.shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
        param_dtype=param_dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
        param_dtype=param_dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
        param_dtype=param_dtype,
    )

  @jax.named_scope("feed_forward")
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    return self.down_proj(activations)


# ---------------------------------------------------------------------------
# Decoder layer (hybrid)
# ---------------------------------------------------------------------------


class DecoderLayer(nnx.Module):
  """Hybrid decoder layer: either full attention or GatedDeltaNet."""

  def __init__(
      self,
      config: "ModelConfig",
      layer_type: str,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    assert layer_type in ("full_attention", "linear_attention")
    self.layer_type = layer_type
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    if layer_type == "full_attention":
      self.token_mixer: Attention | GatedDeltaNet = Attention(
          config=config,
          rngs=rngs,
          param_dtype=config.param_dtype,
      )
    else:
      self.token_mixer = GatedDeltaNet(
          config=config,
          rngs=rngs,
          param_dtype=config.param_dtype,
      )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    self.mlp = MLP(
        config=config,
        rngs=rngs,
        param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      padding_mask: jaxtyping.Array | None = None,  # [B, T] — for linear layers
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)

    if self.layer_type == "full_attention":
      new_cache, token_output = self.token_mixer(
          inputs_normalized, segment_pos, cache, attn_mask
      )
    else:
      new_cache, token_output = self.token_mixer(
          inputs_normalized, cache, padding_mask
      )

    x = x + token_output
    residual = x
    x = self.post_attention_layernorm(x)
    x = self.mlp(x)
    return new_cache, residual + x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Qwen3_5(BackendMappingMixin, nnx.Module):
  """Qwen3.5 hybrid language model (text-only or multimodal)."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(
            config=config,
            layer_type=config.layer_types[i],
            rngs=rngs,
            shd_config=shd_config,
        )
        for i in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str="BTD,DV->BTV",
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=shd_config.emb_dv,
          param_dtype=config.param_dtype,
      )
    self.visual: VisionModel | None = (
        VisionModel(config.vision_config, rngs=rngs)
        if config.vision_config
        else None
    )

  # ------------------------------------------------------------------
  # Cache initialisation
  # ------------------------------------------------------------------

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initialises per-layer caches for both attention and linear layers."""
    config = self.config
    cache: Cache = {}
    for i, layer_type in enumerate(config.layer_types):
      name = f"layer_{i}"
      if layer_type == "full_attention":
        shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
        cache[name] = {
            "k": jnp.zeros(shape, dtype=dtype),
            "v": jnp.zeros(shape, dtype=dtype),
            "end_index": jnp.zeros((batch_size,), dtype=jnp.int32),
        }
      else:  # linear_attention
        conv_dim = (
            config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        cache[name] = {
            "conv_state": jnp.zeros(
                (batch_size, config.linear_conv_kernel_dim - 1, conv_dim),
                dtype=dtype,
            ),
            "recurrent_state": jnp.zeros(
                (
                    batch_size,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ),
                dtype=dtype,
            ),
        }
    return cache

  # ------------------------------------------------------------------
  # Forward pass
  # ------------------------------------------------------------------

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [3, B, L]  3D M-RoPE
      pixel_values: jaxtyping.Array | None,
      vision_grid: VisionGridData | None,
      cache: Cache | None,
      input_mask: jaxtyping.Array | None,  # [B, L] padding (1=real, 0=pad)
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Forward pass.

    Args:
      input_tokens: Token ids, shape [B, L].
      positions: 3D M-RoPE position ids [3, B, L] from ``get_rope_index``.
        Row 0 is the text/temporal axis used for causal masking.
      pixel_values: Flat patch tokens [N_patches, patch_volume] or None.
      vision_grid: Pre-computed vision grid data or None.
      cache: Per-layer KV / recurrent state cache or None.
      input_mask: Padding mask [B, L] (1 = real, 0 = pad) or None.
      output_hidden_states: If True, sow the final hidden states.

    Returns:
      ``(logits [B, L, vocab_size], new_cache)``
    """
    new_cache: Cache | None = None if cache is None else {}
    x = self.embedder.encode(input_tokens)
    bsz = x.shape[0]

    # Build causal attention mask from the temporal M-RoPE axis.
    text_positions = positions[0]  # [B, L]
    causal_mask = make_causal_mask_from_positions(text_positions, input_mask)

    # When using a KV cache the key dimension equals cache_size, not seq_len.
    # The attention logit tensor has shape [B, heads, L_q, cache_size], so
    # the mask must match [B, L_q, cache_size].
    #
    # Two cases:
    #   Prefill  (L_q > 1): pad the [B, L, L] causal mask with False on the
    #     right — those cache slots have never been written.
    #   Decode   (L_q = 1): the position-based mask is [B, 1, 1] and zero-
    #     padding would incorrectly block all slots except position 0.
    #     Instead build a validity mask: attend to every slot that was already
    #     written (indices 0 … end_index, inclusive, because the attention
    #     block writes the current token *before* computing attention).
    if cache is not None:
      first_attn_cache = next((v for v in cache.values() if "k" in v), None)
      if first_attn_cache is not None:
        cache_size = first_attn_cache["k"].shape[1]
        seq_len_q = causal_mask.shape[1]
        if seq_len_q > 1:
          # Prefill: extend key dimension from seq_len to cache_size.
          pad = cache_size - causal_mask.shape[-1]
          if pad > 0:
            causal_mask = jnp.pad(causal_mask, ((0, 0), (0, 0), (0, pad)))
        else:
          # Decode: attend to all slots 0 … end_index (written before attn).
          end_index = first_attn_cache["end_index"][0]
          valid = (
              jnp.arange(cache_size, dtype=jnp.int32)[None, None, :]
              <= end_index
          )  # [1, 1, cache_size]
          bsz = causal_mask.shape[0]
          causal_mask = jnp.broadcast_to(valid, (bsz, 1, cache_size))

    # Inject vision tokens if present.
    if self.config.vision_config is not None and pixel_values is not None:
      image_pad_id = self.config.vision_config.image_pad_id
      vision_embeds = self._encode_vision(pixel_values, vision_grid)
      vision_embeds = vision_embeds.cast(
          self.config.param_dtype
      ).with_batch_dim(bsz)

      def _inject(h, tok, vis):
        num_vis = vis.shape[0]
        pos = jnp.where(
            tok == jnp.int32(image_pad_id), size=num_vis, fill_value=-1
        )[0]
        valid = pos >= 0
        pos = jnp.where(valid, pos, 0)
        updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[pos])
        return h.at[pos].set(updates)

      x = jax.vmap(_inject)(x, input_tokens, vision_embeds.tokens)

    for i, layer in enumerate(self.layers):
      layer_name = f"layer_{i}"
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          causal_mask,
          input_mask,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, "all_hidden_states", x)

    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)

    return logits, new_cache  # pytype: disable=bad-return-type

  def _encode_vision(
      self,
      pixel_values: jax.Array,
      precomputed: VisionGridData,
  ) -> VisionEmbeddings:
    if self.visual is None:
      raise ValueError("Vision backbone not configured.")
    tokens, deepstack = self.visual(pixel_values, precomputed)
    return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))

  def get_model_input(self):
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        "input_tokens": jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        "positions": jnp.ones(
            (3, dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        "pixel_values": None,
        "vision_grid": None,
        "cache": None,
        "input_mask": jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.bool_
        ),
    }
