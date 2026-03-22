"""JAX implementation of the fused recurrent gated delta rule.

Interface matches flash-linear-attention (fla):
  fused_recurrent_gated_delta_rule(q, k, v, g, beta, ...) -> (o, ht)
  chunk_gated_delta_rule(...)                              -> delegates here

Both functions accept:
  q, k :         [B, T, H, K]
  v :            [B, T, H, V]
  g :            [B, T, H]     log-space decay (negative)
  beta :         [B, T, H]
  initial_state: [B, H, K, V] or None
  output_final_state: bool
  scale:         float or None  (defaults to K**-0.5)
  use_qk_l2norm_in_kernel: bool (apply L2-norm; default False)

Returns (o [B, T, H, V], ht [B, H, K, V] or None).

Memory profile
--------------
The backward uses sqrt(T) gradient checkpointing.  The sequence is split
into ceil(sqrt(T)) chunks of size ceil(sqrt(T)).  One hidden state is saved
at the start of each chunk during the forward pass.  During backward, each
chunk's internal states are recomputed exactly from its checkpoint, then the
standard VJP is applied within the chunk.

Peak extra memory per layer:
  forward  : O(sqrt(T)) checkpoint states  +  O(T·H·V) outputs (unavoidable)
  backward : O(sqrt(T)) checkpoint states  +  O(sqrt(T)·H·K·V) per chunk recompute
             +  O(T·H·K) gradient outputs (unavoidable)

This gives numerically exact gradients for any sequence length.  The
Sherman-Morrison inversion approach (O(1) memory, no checkpoints) is
numerically unstable for long sequences because dividing by the decay factor
at each backward step amplifies reconstruction errors exponentially.
"""

from __future__ import annotations

import math

import jax
import jax.lax as lax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Per-step function
# ---------------------------------------------------------------------------


def _step(state, inputs):
  """One forward step of the gated delta rule.

  Args:
    state:  [B, H, K, V]  float32
    inputs: (q_t, k_t, v_t, g_t, b_t) each [B, H, K/V/1]

  Returns:
    (new_state [B,H,K,V], out_t [B,H,V])  both float32
  """
  q_t, k_t, v_t, g_t, b_t = inputs
  state = state.astype(jnp.float32)
  decay = jnp.exp(g_t)[:, :, jnp.newaxis, jnp.newaxis]  # [B,H,1,1]
  state = state * decay
  kv = jnp.einsum("bhd,bhde->bhe", k_t.astype(jnp.float32), state)  # [B,H,V]
  delta = (v_t.astype(jnp.float32) - kv) * b_t[:, :, jnp.newaxis]  # [B,H,V]
  state = state + jnp.einsum(
      "bhd,bhe->bhde", k_t.astype(jnp.float32), delta
  )  # [B,H,K,V]
  out_t = jnp.einsum("bhd,bhde->bhe", q_t.astype(jnp.float32), state)  # [B,H,V]
  return state, out_t


# ---------------------------------------------------------------------------
# Custom-VJP scan with sqrt(T) gradient checkpointing
# ---------------------------------------------------------------------------


@jax.custom_vjp
def _scan_op(init, xs):
  """Forward scan: (init [B,H,K,V], xs tuple[T,B,H,*]) -> (final, ys [T,B,H,V])."""
  return lax.scan(_step, init, xs)


def _scan_op_fwd(init, xs):
  T = xs[0].shape[0]
  C = max(1, int(math.ceil(T**0.5)))  # chunk size ≈ sqrt(T)
  pad = (-T) % C  # pad so T+pad is divisible by C
  n_chunks = (T + pad) // C

  xs_padded = xs
  if pad > 0:
    xs_padded = tuple(
        jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), x.dtype)], axis=0)
        for x in xs
    )

  # Reshape to [n_chunks, C, B, H, ...]
  xs_chunked = tuple(x.reshape(n_chunks, C, *x.shape[1:]) for x in xs_padded)

  def scan_chunk(state, xs_c):
    checkpoint = state  # save state at chunk start
    new_state, ys_c = lax.scan(_step, state, xs_c)
    return new_state, (ys_c, checkpoint)

  final_state, (ys_chunked, checkpoints) = lax.scan(
      scan_chunk, init, xs_chunked
  )

  # checkpoints: [n_chunks, B, H, K, V]
  # ys_chunked:  [n_chunks, C, B, H, V] -> unpad -> [T, B, H, V]
  ys = ys_chunked.reshape(n_chunks * C, *ys_chunked.shape[2:])[:T]

  return (final_state, ys), (xs_padded, checkpoints, final_state)


def _scan_op_bwd(residuals, g_out):
  xs_padded, checkpoints, final_state = residuals
  d_final, d_outputs = g_out  # [B,H,K,V], [T_orig,B,H,V]

  T_orig = d_outputs.shape[0]
  T_padded = xs_padded[0].shape[0]
  n_chunks = checkpoints.shape[0]
  C = T_padded // n_chunks

  # Pad d_outputs to T_padded
  if T_padded > T_orig:
    pad = T_padded - T_orig
    d_outputs = jnp.concatenate(
        [d_outputs, jnp.zeros((pad, *d_outputs.shape[1:]), d_outputs.dtype)],
        axis=0,
    )

  xs_chunked = tuple(x.reshape(n_chunks, C, *x.shape[1:]) for x in xs_padded)
  do_chunked = d_outputs.reshape(n_chunks, C, *d_outputs.shape[1:])

  def chunk_bwd(d_state, chunk_data):
    xs_c, do_c, checkpoint_c = chunk_data
    # xs_c:        tuple of [C, B, H, ...]
    # do_c:        [C, B, H, V]
    # checkpoint_c:[B, H, K, V]  state at the start of this chunk

    # Recompute the C forward states exactly from the checkpoint.
    def fwd_only(state, inputs_t):
      new_state, _ = _step(state, inputs_t)
      return new_state, new_state  # carry=new_state, output=new_state

    _, states_after = lax.scan(fwd_only, checkpoint_c, xs_c)
    # states_after[t] = state AFTER step t  shape: [C, B, H, K, V]

    # State BEFORE step t: checkpoint for t=0, states_after[t-1] for t>0.
    states_before = jnp.concatenate(
        [checkpoint_c[None], states_after[:-1]], axis=0
    )  # [C, B, H, K, V]

    # Exact VJP backward through this chunk (reverse order).
    def bwd_step(d_s, data):
      s_t, inputs_t, do_t = data
      _, vjp_fn = jax.vjp(_step, s_t, inputs_t)
      d_s_new, d_inputs_t = vjp_fn((d_s, do_t))
      return d_s_new, d_inputs_t

    d_checkpoint, d_inputs_c = lax.scan(
        bwd_step, d_state, (states_before, xs_c, do_c), reverse=True
    )
    return d_checkpoint, d_inputs_c

  # Process chunks in reverse order (last chunk first).
  d_init, d_inputs_chunked = lax.scan(
      chunk_bwd,
      d_final,
      (xs_chunked, do_chunked, checkpoints),
      reverse=True,
  )

  # d_inputs_chunked: tuple of [n_chunks, C, ...] -> [T_padded, ...] -> [:T_orig]
  d_inputs = tuple(
      di.reshape(n_chunks * C, *di.shape[2:])[:T_orig]
      for di in d_inputs_chunked
  )

  return d_init, d_inputs


_scan_op.defvjp(_scan_op_fwd, _scan_op_bwd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g=None,
    beta=None,
    scale=None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    # unused fla compat kwargs:
    gk=None,
    gv=None,
    cu_seqlens=None,
    transpose_state_layout: bool = False,
):
  """Fused recurrent gated delta rule — JAX implementation.

  Args:
    q, k : [B, T, H, K]  queries / keys
    v    : [B, T, H, V]  values
    g    : [B, T, H]     log-space decay (negative); required
    beta : [B, T, H]     write strength in (0, 1); required
    scale: float or None  key scaling; defaults to K**-0.5
    initial_state: [B, H, K, V] or None
    output_final_state: bool

  Returns:
    o  : [B, T, H, V]
    ht : [B, H, K, V] if output_final_state else None
  """
  if g is None:
    raise ValueError("g (log-decay) is required")
  if beta is None:
    raise ValueError("beta is required")
  if gk is not None or gv is not None:
    raise NotImplementedError("per-key/value decay (gk/gv) not supported")
  if cu_seqlens is not None:
    raise NotImplementedError("variable-length sequences not supported")

  B, T, H, K = q.shape
  if scale is None:
    scale = float(K**-0.5)

  if use_qk_l2norm_in_kernel:
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
  q = q * scale

  h0 = (
      jnp.zeros((B, H, K, v.shape[-1]), dtype=jnp.float32)
      if initial_state is None
      else initial_state.astype(jnp.float32)
  )

  # Transpose T to leading axis for scan: [T, B, H, *]
  xs = (
      jnp.moveaxis(q, 1, 0),
      jnp.moveaxis(k, 1, 0),
      jnp.moveaxis(v, 1, 0),
      jnp.moveaxis(g, 1, 0),
      jnp.moveaxis(beta, 1, 0),
  )

  ht, ys = _scan_op(h0, xs)
  # ys: [T, B, H, V] -> [B, T, H, V]
  o = jnp.moveaxis(ys, 0, 1).astype(v.dtype)
  return o, (ht if output_final_state else None)


def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
  """Chunked gated delta rule — delegates to fused_recurrent."""
  return fused_recurrent_gated_delta_rule(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      **kwargs,
  )
