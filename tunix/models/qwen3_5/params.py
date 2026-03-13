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

"""Utils for loading and converting Qwen3.5 PT weights."""

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen3_5 import model as model_lib


def _squeeze_conv1d_weight(
    params: dict[str, jax.Array],
) -> dict[str, jax.Array]:
  """Reshape Conv1d weights from PyTorch layout to our layout.

  PyTorch stores depthwise Conv1d weights as ``[C, 1, K]``.  We use
  ``[K, C]`` so that the simple depthwise computation
  ``sum(window * weight, axis=kernel_axis)`` works directly.
  """
  updated = dict(params)
  # After key mapping the weight lives at 'layers.N.token_mixer.conv1d_weight'.
  pattern = re.compile(r'.*token_mixer\.conv1d_weight')
  for key in list(params.keys()):
    if pattern.match(key):
      w = params[key]  # [C, 1, K]
      w = w.squeeze(1)  # [C, K]
      updated[key] = w.T  # [K, C]
  return updated


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Returns a mapping from HuggingFace safetensors keys to tunix keys.

  Each entry maps a regex pattern for a HF key to ``(tunix_key_pattern,
  (permute_axes, reshape_shape))``.  Use ``None`` for the transform when no
  reshaping is needed.
  """
  # Common attention block mappings shared by full-attention layers.
  attn_mappings = {
      # q_proj: note 2× head_dim compared with standard Qwen3.
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight': (
          r'layers.\1.token_mixer.q_proj.w',
          (
              (1, 0),
              (cfg.embed_dim, cfg.num_heads, cfg.head_dim * 2),
          ),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight': (
          r'layers.\1.token_mixer.k_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight': (
          r'layers.\1.token_mixer.v_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight': (
          r'layers.\1.token_mixer.o_proj.w',
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight': (
          r'layers.\1.token_mixer.q_norm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight': (
          r'layers.\1.token_mixer.k_norm.w',
          None,
      ),
  }

  # Linear-attention (GatedDeltaNet) layer mappings.
  linear_attn_mappings = {
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_qkv\.weight': (
          r'layers.\1.token_mixer.in_proj_qkv.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_z\.weight': (
          r'layers.\1.token_mixer.in_proj_z.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_b\.weight': (
          r'layers.\1.token_mixer.in_proj_b.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.in_proj_a\.weight': (
          r'layers.\1.token_mixer.in_proj_a.kernel',
          ((1, 0), None),
      ),
      # conv1d weight: already transposed by _squeeze_conv1d_weight to [K, C].
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.conv1d\.weight': (
          r'layers.\1.token_mixer.conv1d_weight',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.dt_bias': (
          r'layers.\1.token_mixer.dt_bias',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.A_log': (
          r'layers.\1.token_mixer.A_log',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.norm\.weight': (
          r'layers.\1.token_mixer.norm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.linear_attn\.out_proj\.weight': (
          r'layers.\1.token_mixer.out_proj.kernel',
          ((1, 0), None),
      ),
  }

  # Shared MLP and norm mappings.
  shared_mappings = {
      r'model\.language_model\.embed_tokens\.weight': (
          'embedder.input_embedding',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight': (
          r'layers.\1.mlp.gate_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.up_proj\.weight': (
          r'layers.\1.mlp.up_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.layers\.([0-9]+)\.mlp\.down_proj\.weight': (
          r'layers.\1.mlp.down_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.language_model\.norm\.weight': ('final_norm.w', None),
      r'model\.language_model\.layers\.([0-9]+)\.input_layernorm\.weight': (
          r'layers.\1.input_layernorm.w',
          None,
      ),
      r'model\.language_model\.layers\.([0-9]+)\.post_attention_layernorm\.weight': (
          r'layers.\1.post_attention_layernorm.w',
          None,
      ),
      r'lm_head\.weight': ('lm_head.w', ((1, 0), None)),
  }

  # Vision encoder mappings (identical to Qwen3-VL, without deepstack).
  v_cfg = cfg.vision_config
  if v_cfg is not None:
    pixel_volume = (
        v_cfg.temporal_patch_size * v_cfg.patch_size**2 * v_cfg.in_channels
    )
    vision_mappings = {
        r'model\.visual\.patch_embed\.proj\.weight': (
            r'visual.patch_embed.proj.kernel',
            ((1, 2, 3, 4, 0), (pixel_volume, v_cfg.hidden_size)),
        ),
        r'model\.visual\.patch_embed\.proj\.bias': (
            r'visual.patch_embed.proj.bias',
            None,
        ),
        r'model\.visual\.pos_embed\.weight': (
            r'visual.pos_embed.embedding',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.weight': (
            r'visual.blocks.\1.attn.qkv_proj.kernel',
            ((1, 0), None),
        ),
        r'model\.visual\.blocks\.([0-9]+)\.attn\.qkv\.bias': (
            r'visual.blocks.\1.attn.qkv_proj.bias',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.attn\.proj\.weight': (
            r'visual.blocks.\1.attn.out_proj.kernel',
            ((1, 0), None),
        ),
        r'model\.visual\.blocks\.([0-9]+)\.attn\.proj\.bias': (
            r'visual.blocks.\1.attn.out_proj.bias',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc1\.weight': (
            r'visual.blocks.\1.mlp.linear1.kernel',
            ((1, 0), None),
        ),
        r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc1\.bias': (
            r'visual.blocks.\1.mlp.linear1.bias',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc2\.weight': (
            r'visual.blocks.\1.mlp.linear2.kernel',
            ((1, 0), None),
        ),
        r'model\.visual\.blocks\.([0-9]+)\.mlp\.linear_fc2\.bias': (
            r'visual.blocks.\1.mlp.linear2.bias',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.norm1\.weight': (
            r'visual.blocks.\1.norm1.scale',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.norm1\.bias': (
            r'visual.blocks.\1.norm1.bias',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.norm2\.weight': (
            r'visual.blocks.\1.norm2.scale',
            None,
        ),
        r'model\.visual\.blocks\.([0-9]+)\.norm2\.bias': (
            r'visual.blocks.\1.norm2.bias',
            None,
        ),
        r'model\.visual\.merger\.linear_fc1\.weight': (
            r'visual.merger.linear_fc1.kernel',
            ((1, 0), None),
        ),
        r'model\.visual\.merger\.linear_fc1\.bias': (
            r'visual.merger.linear_fc1.bias',
            None,
        ),
        r'model\.visual\.merger\.linear_fc2\.weight': (
            r'visual.merger.linear_fc2.kernel',
            ((1, 0), None),
        ),
        r'model\.visual\.merger\.linear_fc2\.bias': (
            r'visual.merger.linear_fc2.bias',
            None,
        ),
        r'model\.visual\.merger\.norm\.weight': (
            r'visual.merger.norm.scale',
            None,
        ),
        r'model\.visual\.merger\.norm\.bias': (
            r'visual.merger.norm.bias',
            None,
        ),
    }
  else:
    vision_mappings = {}

  return {
      **shared_mappings,
      **attn_mappings,
      **linear_attn_mappings,
      **vision_mappings,
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen3_5:
  """Load tensors from a safetensors directory and create a Qwen3.5 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3_5,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_squeeze_conv1d_weight,
      dtype=dtype,
  )


def _state_key_to_safetensors_key(name: str) -> str:
  """Map a tunix layer path back to a HuggingFace safetensors key.

  Used when saving a LoRA-merged checkpoint in HF format.
  """
  # token_mixer.{q,k,v,o}_proj  →  self_attn.{q,k,v,o}_proj
  name = name.replace('.token_mixer.q_proj', '.self_attn.q_proj')
  name = name.replace('.token_mixer.k_proj', '.self_attn.k_proj')
  name = name.replace('.token_mixer.v_proj', '.self_attn.v_proj')
  name = name.replace('.token_mixer.o_proj', '.self_attn.o_proj')
  return f'model.{name}.weight'


_HUGGINGFACE_TRANSPOSE_RULES = {
    'q_proj': (1, 0),
    'k_proj': (1, 0),
    'v_proj': (1, 0),
    'o_proj': (1, 0),
    'up_proj': (1, 0),
    'down_proj': (1, 0),
    'gate_proj': (1, 0),
}


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen3_5,
    rank: int,
    alpha: float,
):
  """Saves a Qwen3.5 model with LoRA weights merged in safetensors format."""
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_state_key_to_safetensors_key,
      transpose_rules=_HUGGINGFACE_TRANSPOSE_RULES,
  )
