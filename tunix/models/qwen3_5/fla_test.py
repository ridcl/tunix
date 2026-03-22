"""Tests for tunix/models/qwen3_5/fla.py.

Three levels:
  1. Forward correctness  — fla output matches _gated_delta_rule reference.
  2. Gradient correctness — fla grads match reference grads (finite-diff check).
  3. Smoke test           — no OOM on a long sequence (T=4096).

Run:
    /venv/bin/python -m tunix.models.qwen3_5.fla_test
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.qwen3_5 import fla as fla_lib
from tunix.models.qwen3_5.model import _gated_delta_rule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2norm(x):
  return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def make_inputs(B, T, H, K, V, seed=0, dtype=jnp.bfloat16):
  """Generate random inputs matching the model's pre-normalised convention."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 6)
  q = _l2norm(jax.random.normal(keys[0], (B, T, H, K))).astype(dtype)
  k = _l2norm(jax.random.normal(keys[1], (B, T, H, K))).astype(dtype)
  v = jax.random.normal(keys[2], (B, T, H, V)).astype(dtype)
  g = -jax.nn.softplus(jax.random.normal(keys[3], (B, T, H))).astype(dtype)
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H))).astype(dtype)
  h0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
  return q, k, v, g, beta, h0


def ref_forward(q, k, v, g, beta, h0):
  """Reference implementation via _gated_delta_rule.

  _gated_delta_rule applies l2norm + scale internally, so we pass raw
  (already l2-normed) inputs and scale=1 by setting scale *before* calling:
  the function multiplies query by scale after its own l2norm.  To compare
  fairly we pass pre-normed q/k with scale=1 by calling the internal helper
  with scale already folded in (query *= scale is the first thing it does).

  Simpler: just call _gated_delta_rule with the same inputs and rely on the
  fact that both implementations apply l2norm and scale identically.
  """
  # _gated_delta_rule always applies _l2norm + scale internally.
  # fla uses use_qk_l2norm_in_kernel=True to match.
  scale = q.shape[-1] ** -0.5
  # Undo our pre-normalisation so _gated_delta_rule sees unnormed inputs:
  # Actually the simplest: just pass the same unnormed inputs to both.
  # We'll generate unnormed inputs for the correctness test.
  o, ht = _gated_delta_rule(q, k, v, g, beta, h0)
  return o, ht


# ---------------------------------------------------------------------------
# Test 1: forward correctness
# ---------------------------------------------------------------------------


def test_forward(tol_rtol=1e-2, tol_atol=1e-2):
  """Compare fla forward against _gated_delta_rule on small inputs."""
  print("Test 1: forward correctness … ", end="", flush=True)

  B, T, H, K, V = 1, 32, 2, 16, 16
  keys = jax.random.split(jax.random.PRNGKey(42), 6)
  # Use unnormed inputs — both implementations apply l2norm internally.
  q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32)
  k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32)
  v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32)
  g = -jax.nn.softplus(jax.random.normal(keys[3], (B, T, H))).astype(
      jnp.float32
  )
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H))).astype(
      jnp.float32
  )
  h0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  # Reference
  o_ref, ht_ref = _gated_delta_rule(q, k, v, g, beta, h0)

  # FLA — l2norm and scale applied inside kernel
  scale = float(K**-0.5)
  o_fla, ht_fla = fla_lib.fused_recurrent_gated_delta_rule(
      q,
      k,
      v,
      g,
      beta,
      scale=scale,
      initial_state=h0,
      output_final_state=True,
      use_qk_l2norm_in_kernel=True,
  )

  o_ref = np.array(o_ref, dtype=np.float32)
  o_fla = np.array(o_fla, dtype=np.float32)
  ht_ref = np.array(ht_ref, dtype=np.float32)
  ht_fla = np.array(ht_fla, dtype=np.float32)

  max_o = np.abs(o_ref - o_fla).max()
  max_ht = np.abs(ht_ref - ht_fla).max()

  if max_o > tol_atol or max_ht > tol_atol:
    print(f"FAIL  (max Δo={max_o:.4f}, max Δht={max_ht:.4f})")
    return False
  print(f"PASS  (max Δo={max_o:.2e}, max Δht={max_ht:.2e})")
  return True


# ---------------------------------------------------------------------------
# Test 2: gradient correctness  (finite-difference check on fla)
# ---------------------------------------------------------------------------


def test_gradients():
  """Check that fla gradients match finite differences."""
  print("Test 2: gradient correctness … ", end="", flush=True)

  B, T, H, K, V = 1, 8, 1, 4, 4
  keys = jax.random.split(jax.random.PRNGKey(7), 6)
  q = jax.random.normal(keys[0], (B, T, H, K))
  k = jax.random.normal(keys[1], (B, T, H, K))
  v = jax.random.normal(keys[2], (B, T, H, V))
  g = -jax.nn.softplus(jax.random.normal(keys[3], (B, T, H)))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))
  h0 = jnp.zeros((B, H, K, V))
  scale = float(K**-0.5)

  def loss(q, k, v, g, beta, h0):
    o, _ = fla_lib.fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=h0,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return jnp.sum(o)

  try:
    jax.test_util.check_grads(
        loss,
        (q, k, v, g, beta, h0),
        order=1,
        modes=["rev"],
        atol=1e-2,
        rtol=1e-2,
    )
    print("PASS")
    return True
  except Exception as e:
    print(f"FAIL  ({e})")
    return False


# ---------------------------------------------------------------------------
# Test 3: fla gradients match finite differences at longer sequence length
# ---------------------------------------------------------------------------


def test_grad_vs_ref():
  """Check fla grad matches finite differences at T=64 (stress test for stability)."""
  print("Test 3: fla grad vs finite-diff (T=64) … ", end="", flush=True)

  B, T, H, K, V = 1, 64, 2, 8, 8
  keys = jax.random.split(jax.random.PRNGKey(13), 5)
  q = jax.random.normal(keys[0], (B, T, H, K))
  k = jax.random.normal(keys[1], (B, T, H, K))
  v = jax.random.normal(keys[2], (B, T, H, V))
  g = -jax.nn.softplus(jax.random.normal(keys[3], (B, T, H)))
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H)))
  h0 = jnp.zeros((B, H, K, V))
  scale = float(K**-0.5)

  def loss(q, k, v, g, beta):
    o, _ = fla_lib.fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=h0,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return jnp.sum(o)

  try:
    jax.test_util.check_grads(
        loss, (q, k, v, g, beta), order=1, modes=["rev"], atol=1e-2, rtol=1e-2
    )
    print("PASS")
    return True
  except Exception as e:
    print(f"FAIL  ({e})")
    return False


# ---------------------------------------------------------------------------
# Test 4: smoke test — no OOM on long sequence
# ---------------------------------------------------------------------------


def test_long_sequence(T=4096):
  """Verify no OOM and outputs are finite for T=4096."""
  print(f"Test 4: long sequence T={T} smoke test … ", end="", flush=True)

  B, H, K, V = 1, 32, 128, 128
  keys = jax.random.split(jax.random.PRNGKey(99), 5)
  q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.bfloat16)
  k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.bfloat16)
  v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.bfloat16)
  g = (-jax.nn.softplus(jax.random.normal(keys[3], (B, T, H)))).astype(
      jnp.bfloat16
  )
  beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H))).astype(
      jnp.bfloat16
  )
  h0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)

  def loss(q, k, v, g, beta):
    o, _ = fla_lib.fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return jnp.sum(o.astype(jnp.float32))

  try:
    val, grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(
        q, k, v, g, beta
    )
    val.block_until_ready()
    ok = jnp.isfinite(val).item()
    if ok:
      print(f"PASS  (loss={float(val):.4f})")
    else:
      print(f"FAIL  (non-finite loss: {val})")
    return ok
  except Exception as e:
    print(f"FAIL  ({e})")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  print(f"JAX devices: {jax.devices()}\n")

  results = [
      test_forward(),
      test_gradients(),
      test_grad_vs_ref(),
      test_long_sequence(),
  ]

  n_pass = sum(results)
  n_fail = len(results) - n_pass
  print(f"\n{n_pass}/{len(results)} passed", end="")
  if n_fail:
    print(f"  ({n_fail} FAILED)")
    sys.exit(1)
  else:
    print()
