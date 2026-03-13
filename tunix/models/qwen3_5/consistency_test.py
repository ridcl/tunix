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

"""Layer-by-layer consistency test: JAX Qwen3.5 vs HuggingFace PyTorch.

Loads the same checkpoint into both frameworks, feeds an identical input, and
compares hidden states after every decoder layer as well as the final logits.

Usage::

    # HuggingFace model ID (downloaded automatically):
    python -m tunix.models.qwen3_5.consistency_test --model_id_or_dir Qwen/Qwen3.5-9B-Instruct

    # Local checkpoint directory:
    python -m tunix.models.qwen3_5.consistency_test --model_id_or_dir /path/to/checkpoint

The script prints a diff table and exits with code 0 if the top-1 prediction
at the last token position matches across both frameworks.
"""

import argparse
import io
import os

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from tunix.models.qwen3_5 import model as model_lib
from tunix.models.qwen3_5 import params as params_lib
from tunix.models.qwen3vl.model import get_rope_index
from tunix.models.qwen3vl.model import make_causal_mask_from_positions

# ---------------------------------------------------------------------------
# Model ID / directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(model_id_or_dir: str) -> str:
  """Return a local directory path for the given model ID or local path.

  If ``model_id_or_dir`` is an existing directory it is returned as-is.
  Otherwise it is treated as a HuggingFace Hub repo ID and the snapshot is
  downloaded (or retrieved from the local cache) via ``huggingface_hub``.
  """
  if os.path.isdir(model_id_or_dir):
    return model_id_or_dir
  print(f'Downloading snapshot for "{model_id_or_dir}" from HuggingFace Hub…')
  return huggingface_hub.snapshot_download(model_id_or_dir)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_jax(t: torch.Tensor | None) -> jnp.ndarray | None:
  """Convert a PyTorch tensor to a JAX array, preserving dtype."""
  if t is None:
    return None
  dtype_map = {
      torch.bfloat16: jnp.bfloat16,
      torch.float32: jnp.float32,
      torch.int32: jnp.int32,
      torch.int64: jnp.int64,
      torch.bool: jnp.bool_,
  }
  if t.dtype == torch.bool:
    return jnp.array(t.detach().cpu().numpy())
  return jnp.array(t.detach().cpu().float().numpy()).astype(dtype_map[t.dtype])


def to_torch(
    x: jnp.ndarray | None,
    device: torch.device | str = 'cpu',
) -> torch.Tensor | None:
  """Convert a JAX array to a PyTorch tensor, preserving dtype."""
  if x is None:
    return None
  dtype_map = {
      'bfloat16': torch.bfloat16,
      'float32': torch.float32,
      'int32': torch.int32,
      'int64': torch.int64,
  }
  return (
      torch.tensor(np.array(x.astype(jnp.float32)))
      .to(dtype_map.get(x.dtype.name, torch.float32))
      .to(device)
  )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    pt_device: str = 'cuda',
    pt_dtype: torch.dtype = torch.bfloat16,
    jax_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[model_lib.Qwen3_5, Qwen3_5ForConditionalGeneration]:
  """Load JAX and PyTorch models from the same safetensors checkpoint."""
  model_dir = resolve_model_dir(model_id_or_dir)
  pt_model = Qwen3_5ForConditionalGeneration.from_pretrained(
      model_dir,
      torch_dtype=pt_dtype,
      device_map=pt_device,
      attn_implementation='eager',
  )
  pt_model.eval()

  with jax.default_device(jax.devices()[0]):
    jax_model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=None, dtype=jax_dtype
    )
  return jax_model, pt_model


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def _inject_vision_pt(
    pt_model: Qwen3_5ForConditionalGeneration,
    input_ids_pt: torch.Tensor,
    inputs_embeds_pt: torch.Tensor,
    pixel_values_pt: torch.Tensor,
    image_grid_thw_pt: torch.Tensor,
    pt_device: str,
) -> torch.Tensor:
  """Inject image features into PT text embeddings (mirrors HF forward)."""
  image_outputs = pt_model.model.get_image_features(
      pixel_values_pt, image_grid_thw_pt, return_dict=True
  )
  image_embeds = image_outputs.pooler_output
  image_embeds = torch.cat(image_embeds, dim=0).to(
      pt_device, inputs_embeds_pt.dtype
  )
  image_mask, _ = pt_model.model.get_placeholder_mask(
      input_ids_pt, inputs_embeds=inputs_embeds_pt, image_features=image_embeds
  )
  return inputs_embeds_pt.masked_scatter(image_mask, image_embeds)


def _inject_vision_jax(
    jax_model: model_lib.Qwen3_5,
    input_tokens_jax: jnp.ndarray,
    x: jnp.ndarray,
    pixel_values_jax: jnp.ndarray,
    image_grid_thw: np.ndarray,
) -> jnp.ndarray:
  """Inject image features into JAX text embeddings (mirrors Qwen3_5.__call__)."""
  image_pad_id = jax_model.config.vision_config.image_pad_id
  grid_data = jax_model.visual.compute_grid_data(image_grid_thw)
  vision_tokens, _ = jax_model.visual(pixel_values_jax, grid_data)
  # Cast to model dtype and add batch dim.
  vision_tokens = vision_tokens.astype(jax_model.config.param_dtype)
  if vision_tokens.ndim == 2:
    vision_tokens = vision_tokens[None, ...]  # [1, N_patches, D]

  bsz = x.shape[0]
  if vision_tokens.shape[0] == 1 and bsz > 1:
    vision_tokens = jnp.tile(vision_tokens, (bsz, 1, 1))

  def _inject(h, tok, vis):
    num_vis = vis.shape[0]
    pos = jnp.where(
        tok == jnp.int32(image_pad_id), size=num_vis, fill_value=-1
    )[0]
    valid = pos >= 0
    pos = jnp.where(valid, pos, 0)
    updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[pos])
    return h.at[pos].set(updates)

  return jax.vmap(_inject)(x, input_tokens_jax, vision_tokens)


def compare_layerwise(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    prompt: str = 'The quick brown fox jumps over the lazy dog.',
    pt_device: str = 'cuda',
    dtype: str = 'bfloat16',
    image_url: str | None = None,
) -> bool:
  """Run a layer-by-layer hidden-state comparison.

  Args:
    model_id_or_dir: HuggingFace repo ID (e.g. ``'Qwen/Qwen3.5-9B-Instruct'``)
      or path to a local safetensors checkpoint directory.
    config: ``ModelConfig`` for the JAX model (must match the checkpoint).
    prompt: Text prompt to use as input.
    pt_device: PyTorch device string (e.g. ``'cuda'`` or ``'cpu'``).
    dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
    image_url: Optional URL to an image for multimodal testing.

  Returns:
    ``True`` if the top-1 next-token prediction matches across both frameworks.
  """
  pt_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
  jax_dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32

  jax_model, pt_model = load_models(
      model_id_or_dir,
      config,
      pt_device=pt_device,
      pt_dtype=pt_dtype,
      jax_dtype=jax_dtype,
  )
  model_dir = resolve_model_dir(model_id_or_dir)

  # ------------------------------------------------------------------
  # Tokenise (with optional image)
  # ------------------------------------------------------------------
  if image_url is not None:
    from PIL import Image
    import requests

    processor = AutoProcessor.from_pretrained(model_dir)
    if os.path.isfile(image_url):
      image = Image.open(image_url).convert('RGB')
    else:
      image = Image.open(
          io.BytesIO(requests.get(image_url, timeout=30).content)
      ).convert('RGB')
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image_url},
            {'type': 'text', 'text': prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors='pt', padding=True
    )
    pixel_values_pt = inputs['pixel_values'].to(pt_device, pt_dtype)
    image_grid_thw_pt = inputs['image_grid_thw'].to(pt_device)
    image_grid_thw_np = np.array(image_grid_thw_pt.cpu())
    print(f'Image grid (t,h,w): {image_grid_thw_np}')
  else:
    processor = None
    pixel_values_pt = None
    image_grid_thw_pt = None
    image_grid_thw_np = None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(prompt, return_tensors='pt')

  input_ids_pt = inputs['input_ids'].to(pt_device)  # [1, L]
  attention_mask_pt = inputs['attention_mask'].to(pt_device)  # [1, L]
  seq_len = input_ids_pt.shape[1]

  # ------------------------------------------------------------------
  # Build 3D position ids
  # ------------------------------------------------------------------
  if image_url is not None:
    # Use the proper M-RoPE index for multimodal inputs.
    position_ids_pt, _ = pt_model.model.get_rope_index(
        input_ids=input_ids_pt,
        image_grid_thw=image_grid_thw_pt,
        attention_mask=attention_mask_pt,
    )
    position_ids_pt = position_ids_pt.to(pt_device)
  else:
    cache_position = torch.arange(seq_len, device=pt_device)
    position_ids_pt = cache_position.view(1, 1, -1).expand(
        3, input_ids_pt.shape[0], -1
    )

  # JAX positions: convert from PT.
  positions_jax = to_jax(position_ids_pt)  # [3, B, L]
  input_tokens_jax = to_jax(input_ids_pt.int())  # [B, L]
  attention_mask_jax = to_jax(attention_mask_pt.bool())  # [B, L]

  # The text sub-model lives at pt_model.model.language_model for
  # Qwen3_5ForConditionalGeneration.
  pt_text_model = pt_model.model.language_model

  # ------------------------------------------------------------------
  # Embeddings
  # ------------------------------------------------------------------
  x = jax_model.embedder.encode(input_tokens_jax)  # [B, L, D]
  inputs_embeds_pt = pt_text_model.embed_tokens(input_ids_pt)  # [B, L, D]

  emb_diff = jnp.abs(
      x.astype(jnp.float32) - to_jax(inputs_embeds_pt).astype(jnp.float32)
  ).max()
  print(f'Embedding max diff (pre-injection): {float(emb_diff):.6f}')

  # ------------------------------------------------------------------
  # Vision injection (if image provided)
  # ------------------------------------------------------------------
  if image_url is not None:
    pixel_values_jax = to_jax(inputs['pixel_values'].to(torch.float32))

    # Compare raw vision encoder outputs.
    pt_vis_out = pt_model.model.get_image_features(
        pixel_values_pt, image_grid_thw_pt, return_dict=True
    )
    pt_vis_embeds = torch.cat(pt_vis_out.pooler_output, dim=0)

    grid_data = jax_model.visual.compute_grid_data(image_grid_thw_np)
    jax_vis_tokens, _ = jax_model.visual(pixel_values_jax, grid_data)

    vis_diff = jnp.abs(
        jax_vis_tokens.astype(jnp.float32)
        - to_jax(pt_vis_embeds).astype(jnp.float32)
    )
    print(
        f'Vision encoder max diff: {float(vis_diff.max()):.6f}'
        f'  mean: {float(vis_diff.mean()):.6f}'
    )

    # Inject into both embeddings.
    inputs_embeds_pt = _inject_vision_pt(
        pt_model,
        input_ids_pt,
        inputs_embeds_pt,
        pixel_values_pt,
        image_grid_thw_pt,
        pt_device,
    )
    x = _inject_vision_jax(
        jax_model,
        input_tokens_jax,
        x,
        pixel_values_jax,
        image_grid_thw_np,
    )

    emb_diff_post = jnp.abs(
        x.astype(jnp.float32) - to_jax(inputs_embeds_pt).astype(jnp.float32)
    ).max()
    print(f'Embedding max diff (post-injection): {float(emb_diff_post):.6f}')

  # ------------------------------------------------------------------
  # Causal masks
  # ------------------------------------------------------------------
  # JAX boolean causal mask [B, L, L]
  text_positions_jax = positions_jax[0]  # [B, L]
  causal_mask_jax = make_causal_mask_from_positions(
      text_positions_jax, attention_mask_jax
  )

  # Convert JAX boolean mask → PyTorch 4-D additive float mask so that
  # both frameworks use *identical* masking regardless of HF's default
  # is_causal=True path.
  causal_mask_4d_pt = (
      torch.tensor(np.array(causal_mask_jax)).unsqueeze(1).to(pt_device)
  )  # [B, 1, L, L]
  explicit_causal_mask_pt = torch.where(
      causal_mask_4d_pt,
      torch.zeros_like(causal_mask_4d_pt, dtype=inputs_embeds_pt.dtype),
      torch.full_like(
          causal_mask_4d_pt, float('-inf'), dtype=inputs_embeds_pt.dtype
      ),
  )

  # For linear-attention layers the HF model uses None when all tokens are
  # real (no padding mask needed).
  linear_attn_mask_pt = (
      None if torch.all(attention_mask_pt == 1) else attention_mask_pt
  )

  # ------------------------------------------------------------------
  # Pre-compute rotary embeddings (shared across all PT layers)
  # ------------------------------------------------------------------
  position_embeddings_pt = pt_text_model.rotary_emb(
      inputs_embeds_pt, position_ids_pt
  )

  # ------------------------------------------------------------------
  # Layer-by-layer comparison
  # ------------------------------------------------------------------
  pt_hidden = inputs_embeds_pt
  cache_position = torch.arange(seq_len, device=pt_device)
  print(
      f'\n{"Layer":>5}  {"Type":<16}  {"max diff":>10}  {"mean diff":>10}'
      f'  {"worst_seq":>9}  {"worst_dim":>9}'
  )
  print('-' * 70)

  for layer_idx, (jax_layer, pt_layer) in enumerate(
      zip(jax_model.layers, pt_text_model.layers)
  ):
    layer_type = config.layer_types[layer_idx]

    # JAX forward
    _, x = jax_layer(
        x,
        positions_jax,
        None,  # no KV cache
        causal_mask_jax,
        attention_mask_jax,  # padding mask for linear layers
    )

    # PyTorch forward — use the explicit JAX-derived causal mask for
    # full-attention layers so both sides apply exactly the same mask.
    pt_layer_mask = (
        linear_attn_mask_pt
        if layer_type == 'linear_attention'
        else explicit_causal_mask_pt
    )
    pt_hidden = pt_layer(
        pt_hidden,
        position_embeddings=position_embeddings_pt,
        attention_mask=pt_layer_mask,
        position_ids=position_ids_pt,
        past_key_values=None,
        cache_position=cache_position,
    )

    abs_diff = jnp.abs(
        x.astype(jnp.float32) - to_jax(pt_hidden).astype(jnp.float32)
    )
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    worst_seq = int(jnp.argmax(abs_diff.max(axis=-1)))
    worst_dim = int(jnp.argmax(abs_diff[0, worst_seq]))
    print(
        f'{layer_idx:>5}  {layer_type:<16}  {max_diff:>10.4f} '
        f' {mean_diff:>10.6f}  {worst_seq:>9}  {worst_dim:>9}'
    )

  # ------------------------------------------------------------------
  # Final norm
  # ------------------------------------------------------------------
  x_normed = jax_model.final_norm(x)
  pt_hidden_normed = pt_text_model.norm(pt_hidden)

  norm_diff = jnp.abs(
      x_normed.astype(jnp.float32)
      - to_jax(pt_hidden_normed).astype(jnp.float32)
  ).max()
  print(f'\nFinal norm max diff: {float(norm_diff):.6f}')

  # ------------------------------------------------------------------
  # Logits at the last token position only (avoids [B, L, vocab] OOM).
  # ------------------------------------------------------------------
  last = seq_len - 1
  x_last = x_normed[:, last : last + 1, :]  # [B, 1, D]
  pt_last = pt_hidden_normed[:, last : last + 1, :]  # [B, 1, D]

  # When embeddings are tied the JAX model reuses the embedding matrix via
  # embedder.decode(); otherwise it has a dedicated lm_head.
  if config.use_tied_embedding:
    jax_logits = jax_model.embedder.decode(x_last)  # [B, 1, V]
  else:
    jax_logits = jax_model.lm_head(x_last)  # [B, 1, V]
  pt_logits = pt_model.lm_head(pt_last)  # [B, 1, V]

  logit_diff = jnp.abs(
      jax_logits.astype(jnp.float32) - to_jax(pt_logits).astype(jnp.float32)
  )
  print(
      f'Logits  max={float(logit_diff.max()):.4f}'
      f'  mean={float(logit_diff.mean()):.4f}'
  )

  jax_top5 = jnp.argsort(jax_logits[0, 0])[-5:][::-1].tolist()
  pt_top5 = jnp.argsort(to_jax(pt_logits)[0, 0])[-5:][::-1].tolist()
  print(f'JAX top-5 tokens: {jax_top5}')
  print(f' PT top-5 tokens: {pt_top5}')
  top1_match = jax_top5[0] == pt_top5[0]
  print(f'Top-1 match: {top1_match}')
  return top1_match


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _config_from_model_id(
    model_id_or_dir: str,
    vision: bool = False,
) -> model_lib.ModelConfig:
  """Infer a ModelConfig from a HuggingFace repo ID or local directory name."""
  name = os.path.basename(model_id_or_dir.rstrip('/'))
  name_lower = name.lower()
  if '0.8b' in name_lower or '0b8' in name_lower:
    return (
        model_lib.ModelConfig.qwen3_5_0b8_vl()
        if vision
        else model_lib.ModelConfig.qwen3_5_0b8()
    )
  if '4b' in name_lower:
    return (
        model_lib.ModelConfig.qwen3_5_4b_vl()
        if vision
        else model_lib.ModelConfig.qwen3_5_4b()
    )
  if '9b' in name_lower:
    return (
        model_lib.ModelConfig.qwen3_5_9b_vl()
        if vision
        else model_lib.ModelConfig.qwen3_5_9b()
    )
  raise ValueError(
      f'Cannot infer ModelConfig from "{model_id_or_dir}". Pass an explicit'
      ' config or use a model ID containing "0.8B", "4B", or "9B".'
  )


def main(
    model_id_or_dir: str = 'Qwen/Qwen3.5-0.8B',
    prompt: str = 'The quick brown fox jumps over the lazy dog.',
    device: str = 'cuda',
    dtype: str = 'bfloat16',
    image_url: str | None = None,
    config: model_lib.ModelConfig | None = None,
) -> bool:
  """Run the layer-by-layer consistency check.

  Can be called directly from a Python console::

      from tunix.models.qwen3_5.consistency_test import main
      main('Qwen/Qwen3.5-9B-Instruct')           # downloads from Hub
      main('/path/to/local/Qwen3.5-9B-Instruct')  # local directory
      main('Qwen/Qwen3.5-0.8B', image_url='https://...')  # multimodal

  Or from the command line::

      python -m tunix.models.qwen3_5.consistency_test --model_id_or_dir Qwen/Qwen3.5-9B-Instruct

  Args:
    model_id_or_dir: HuggingFace repo ID or local checkpoint directory.  When
      called from the command line this is taken from ``--model_id_or_dir``.
    prompt: Text prompt to tokenise.
    device: PyTorch device string (e.g. ``'cuda'`` or ``'cpu'``).
    dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
    image_url: If set, use the vision-language config and inject this image.
    config: Explicit ``ModelConfig`` to use.  Auto-detected from
      ``model_id_or_dir`` if ``None``.

  Returns:
    ``True`` if the top-1 next-token prediction matches across both frameworks.
  """
  if config is None:
    config = _config_from_model_id(
        model_id_or_dir, vision=image_url is not None
    )
  return compare_layerwise(
      model_id_or_dir=model_id_or_dir,
      config=config,
      prompt=prompt,
      pt_device=device,
      dtype=dtype,
      image_url=image_url,
  )


if __name__ == '__main__' and '__file__' in globals():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--model_id_or_dir',
      default='Qwen/Qwen3.5-0.8B',
      help='HuggingFace repo ID or local checkpoint directory.',
  )
  parser.add_argument(
      '--prompt',
      default='The quick brown fox jumps over the lazy dog.',
      help='Text prompt to tokenise.',
  )
  parser.add_argument(
      '--device',
      default='cuda',
      help='PyTorch device (e.g. "cuda" or "cpu").',
  )
  parser.add_argument(
      '--dtype',
      choices=['bfloat16', 'float32'],
      default='bfloat16',
      help='Compute dtype.',
  )
  parser.add_argument(
      '--image_url',
      default=None,
      help='URL of an image for multimodal (vision-language) testing.',
  )
  _args = parser.parse_args()
  raise SystemExit(
      0
      if main(
          model_id_or_dir=_args.model_id_or_dir,
          prompt=_args.prompt,
          device=_args.device,
          dtype=_args.dtype,
          image_url=_args.image_url,
      )
      else 1
  )
