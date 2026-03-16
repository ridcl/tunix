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

"""Fine-tuning Qwen3.5-4B with PeftTrainer on DocumentVQA (LoRA / QLoRA).

Fine-tunes Qwen/Qwen3.5-4B on the HuggingFaceM4/DocumentVQA dataset using
LoRA or QLoRA via qwix.  Images are included in the training inputs.

Key differences from a text-only fine-tuning example:
  - Uses qwen3_5_4b_vl() model config (includes vision encoder).
  - Each training item carries pixel_values and image_grid_thw alongside
    text tokens.
  - Images are NOT manually resized; the processor"s built-in smart_resize
    handles scaling while preserving the original aspect ratio, capped at
    MAX_PIXELS total pixels.
  - pixel_values are zero-padded to MAX_PATCHES per image so that grain.Batch
    can stack items with consistent shapes even when different images produce
    slightly different patch grids.
  - encode_messages() handles tokenisation, loss masking, and 3-D M-RoPE
    position computation in the data pipeline.
  - VisionGridData (positional embeddings for the visual encoder) is
    precomputed in _Qwen3VLPeftTrainer._prepare_inputs(), which runs OUTSIDE
    the JAX JIT boundary.  VisionModel.compute_grid_data() uses np.asarray()
    and Python-level loops over grid dimensions and cannot be traced by JAX.

Usage::

    python -m tunix.models.qwen3_5.train_example
"""

from __future__ import annotations

import logging
import os
from typing import Any

import datasets
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from transformers import AutoProcessor
from tunix.models.qwen3_5 import model as model_lib
from tunix.models.qwen3_5 import params as params_lib
from tunix.models.qwen3_5.consistency_test import resolve_model_dir
from tunix.models.qwen3_5.utils import encode_messages
from tunix.models.qwen3vl.vision import VisionGridData
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-4B"

# Maximum image pixels passed to the processor"s smart_resize.  The processor
# preserves the image"s original aspect ratio and scales so that H*W ≤
# MAX_PIXELS, rounding H and W to multiples of patch_size.
# With patch_size=16 and temporal_patch_size=2 this bounds each image to at
# most MAX_PIXELS/16² × 2 = 800 patch tokens before spatial merge.
MAX_PIXELS = 320 * 320  # 102 400 pixels

# Upper bound on patch tokens per image (T x H_patches x W_patches ≤ this).
# Used to zero-pad pixel_values so grain.Batch can stack variable-grid images.
_PATCH_SIZE = 16
_TEMPORAL_PATCH_SIZE = 2
MAX_PATCHES = (MAX_PIXELS // (_PATCH_SIZE * _PATCH_SIZE)) * _TEMPORAL_PATCH_SIZE

BATCH_SIZE = 2
MAX_SEQ_LEN = 512  # ≤200 vision tokens (after merge) + text tokens

USE_QUANTIZATION = False  # True -> QLoRA, False -> LoRA
LORA_RANK = 16
LORA_ALPHA = float(2 * LORA_RANK)

MAX_STEPS = 100
EVAL_EVERY_N_STEPS = 20

LORA_CKPT_DIR = "/tmp/qwen3_5_vl_lora_ckpts"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _ProcessItem(grain.MapTransform):
  """Process one DocumentVQA example into training tensors.

  Each item is returned as a plain dict of fixed-shape numpy arrays so that
  grain.Batch can stack them into a batch without issues.

  pixel_values are zero-padded to [MAX_PATCHES, C] so that images with
  different grid sizes (different aspect ratios) can be batched together.
  The actual patch count is recoverable from image_grid_thw: T×H×W.

  Output dict keys (single item, shapes before batching):
    input_tokens:  [MAX_SEQ_LEN]      int32
    input_mask:    [MAX_SEQ_LEN]      bool,  True for answer tokens only
    pixel_values:  [MAX_PATCHES, C]   float32, zero-padded patch tokens
    image_grid_thw:[3]                int32, (T, H_patches, W_patches)
    positions_3d:  [3, MAX_SEQ_LEN]  int32, 3-D M-RoPE positions
  """

  def __init__(
      self,
      processor: AutoProcessor,
      max_seq_len: int,
      max_patches: int,
      vcfg: model_lib.VisionModelConfig,
  ):
    self._processor = processor
    self._max_seq_len = max_seq_len
    self._max_patches = max_patches
    self._vcfg = vcfg

  def map(self, element: dict[str, Any]) -> dict[str, np.ndarray]:
    image = element["image"].convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": element["question"]},
            ],
        },
        {"role": "assistant", "content": element["answers"][0]},
    ]

    batch = encode_messages(
        self._processor,
        [conversation],
        loss_roles={"assistant"},
        max_seq_len=self._max_seq_len,
        vcfg=self._vcfg,
        padding="max_length",
        truncation=True,
    )

    # Pad pixel_values to MAX_PATCHES so grain.Batch can stack items whose
    # images produced different grid sizes (different aspect ratios).
    pv = batch.pixel_values  # [n_patches, C]
    pixel_values = np.zeros((self._max_patches, pv.shape[1]), dtype=pv.dtype)
    pixel_values[: pv.shape[0]] = pv

    return {
        "input_tokens": batch.input_tokens[0],  # [L]
        "input_mask": batch.completion_mask[0],  # [L]
        "pixel_values": pixel_values,  # [MAX_PATCHES, C]
        "image_grid_thw": batch.image_grid_thw[0],  # [3]
        "positions_3d": batch.positions[:, 0, :],  # [3, L]
    }


class _FilterOverlength(grain.FilterTransform):
  """Drop items that were truncated so much that no answer tokens remain."""

  def filter(self, x: dict[str, np.ndarray]) -> bool:
    return bool(x["input_mask"].any())


def create_datasets(
    processor: AutoProcessor,
    vcfg: model_lib.VisionModelConfig,
    batch_size: int,
    max_seq_len: int,
) -> tuple[grain.DataLoader, grain.DataLoader]:
  """Return (train_loader, eval_loader) for HuggingFaceM4/DocumentVQA."""
  hf_ds = datasets.load_dataset("HuggingFaceM4/DocumentVQA")
  train_hf = hf_ds["train"]
  eval_hf = hf_ds["validation"]

  ops = [
      _ProcessItem(processor, max_seq_len, MAX_PATCHES, vcfg),
      _FilterOverlength(),
      grain.Batch(batch_size=batch_size, drop_remainder=True),
  ]

  def _make_loader(hf_split, num_epochs):
    return grain.DataLoader(
        data_source=hf_split,
        sampler=grain.IndexSampler(
            num_records=len(hf_split),
            num_epochs=num_epochs,
            shard_options=grain.NoSharding(),
        ),
        operations=ops,
        # worker_count=0: keep single-threaded; AutoProcessor is not fork-safe.
        worker_count=0,
    )

  return _make_loader(train_hf, num_epochs=3), _make_loader(
      eval_hf, num_epochs=1
  )


# ---------------------------------------------------------------------------
# Model input conversion and loss
# ---------------------------------------------------------------------------


def make_gen_model_input_fn(pad_id: int):
  """Return a gen_model_input_fn that converts a batched dict to model kwargs.

  By the time this function is called (inside the JIT-compiled train step),
  _Qwen3VLPeftTrainer._prepare_inputs() has already:
    - sliced pixel_values from [B, MAX_PATCHES, C] → [B*n_patches, C]
    - replaced image_grid_thw with vision_grid (VisionGridData)

  Remaining shapes at call time:
    input_tokens:      [B, L]
    input_mask:        [B, L]
    pixel_values:      [B*n_patches, C]
    vision_grid:       VisionGridData pytree
    positions_3d:      [B, 3, L]  ← needs transpose to [3, B, L]
  """

  def _fn(batch: dict) -> dict:
    input_tokens = jnp.array(batch["input_tokens"])  # [B, L]
    input_mask = jnp.array(batch["input_mask"])  # [B, L]
    pixel_values = jnp.array(batch["pixel_values"], dtype=jnp.bfloat16)

    # grain.Batch stacks [3, L] items along axis 0 → [B, 3, L]; transpose back.
    positions_3d = jnp.moveaxis(
        jnp.array(batch["positions_3d"]), 1, 0
    )  # [3, B, L]

    pad_mask = input_tokens != pad_id  # [B, L]

    return {
        "input_tokens": input_tokens,
        "input_mask": input_mask,
        "positions": positions_3d,
        "attention_mask": pad_mask.astype(jnp.bool_),
        "pixel_values": pixel_values,
        "vision_grid": batch["vision_grid"],
    }

  return _fn


def loss_fn(
    model: model_lib.Qwen3_5,
    input_tokens: jax.Array,  # [B, L]
    input_mask: jax.Array,  # [B, L]
    positions: jax.Array,  # [3, B, L]
    attention_mask: jax.Array,  # [B, L]
    pixel_values: jax.Array,  # [B*n_patches, C]
    vision_grid: VisionGridData,  # pre-computed outside JIT
) -> jax.Array:
  """Cross-entropy loss over answer tokens only."""
  logits, _ = model(
      input_tokens,
      positions,
      pixel_values,
      vision_grid,
      None,  # cache
      attention_mask,
  )

  # Shift by 1: predict token t+1 from the hidden state at position t.
  logits = logits[:, :-1, :].astype(jnp.float32)  # [B, L-1, V]
  targets = input_tokens[:, 1:]  # [B, L-1]
  mask = input_mask[:, 1:].astype(jnp.float32)  # [B, L-1]

  one_hot = jax.nn.one_hot(targets, logits.shape[-1])
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  token_loss = -jnp.sum(one_hot * log_probs, axis=-1)  # [B, L-1]
  return jnp.sum(token_loss * mask) / jnp.sum(mask)


# ---------------------------------------------------------------------------
# LoRA / QLoRA helper
# ---------------------------------------------------------------------------

_LORA_TARGETS = ".*q_proj|.*k_proj|.*gate_proj|.*up_proj|.*down_proj"


def get_lora_model(
    base_model: model_lib.Qwen3_5,
    rank: int = LORA_RANK,
    alpha: float = LORA_ALPHA,
    quantize: bool = USE_QUANTIZATION,
) -> model_lib.Qwen3_5:
  """Wrap base_model with LoRA (or QLoRA) adapters."""
  lora_provider = qwix.LoraProvider(
      module_path=_LORA_TARGETS,
      rank=rank,
      alpha=alpha,
      **(dict(weight_qtype="nf4", tile_size=128) if quantize else {}),
  )
  model_input = base_model.get_model_input()
  return qwix.apply_lora_to_model(base_model, lora_provider, **model_input)


# ---------------------------------------------------------------------------
# Custom trainer: precompute VisionGridData outside the JIT boundary
# ---------------------------------------------------------------------------


class _Qwen3VLPeftTrainer(peft_trainer.PeftTrainer):
  """PeftTrainer subclass that handles vision preprocessing outside JIT.

  VisionModel.compute_grid_data() uses numpy operations on concrete grid
  values and Python-level loops — it cannot be traced by JAX JIT.  This
  subclass overrides _prepare_inputs (called before train_step is dispatched
  to the JIT-compiled function) to:

    1. Slice the padded pixel_values [B, MAX_PATCHES, C] to the actual patch
       count [B * n_patches, C] determined by image_grid_thw.
    2. Call compute_grid_data with the concrete image_grid_thw to obtain a
       VisionGridData pytree that can be passed into the JIT boundary as a
       regular JAX array pytree.
  """

  def __init__(self, visual_model, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._visual_model = visual_model

  def _prepare_inputs(self, batch):
    image_grid_thw = np.array(batch["image_grid_thw"])  # [B, 3], concrete

    # All images in the batch are assumed to have the same grid (grain.Batch
    # would have raised a shape error otherwise).
    n_patches = int(np.prod(image_grid_thw[0]))  # T × H × W for one image
    B = image_grid_thw.shape[0]

    # Slice padded pixel_values to the actual patch count per image.
    pv = np.array(batch["pixel_values"])  # [B, MAX_PATCHES, C]
    pixel_values = pv[:, :n_patches, :].reshape(B * n_patches, -1)

    # Precompute positional data for the visual encoder (outside JIT).
    vision_grid = self._visual_model.compute_grid_data(
        jnp.array(image_grid_thw)
    )

    batch = dict(batch)
    batch["pixel_values"] = pixel_values  # [B*n_patches, C]
    batch["vision_grid"] = vision_grid
    del batch["image_grid_thw"]
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  os.makedirs(LORA_CKPT_DIR, exist_ok=True)

  # --- Load model ---
  config = model_lib.ModelConfig.qwen3_5_4b_vl()
  model_dir = resolve_model_dir(MODEL_ID)

  logger.info("Loading model from %s", model_dir)
  base_model = params_lib.create_model_from_safe_tensors(
      model_dir, config, mesh=None, dtype=jnp.bfloat16
  )

  # --- Processor (tokenizer + image processor) ---
  processor = AutoProcessor.from_pretrained(model_dir)
  # Cap image size via the processor"s smart_resize (aspect-ratio preserving).
  # processor.image_processor.max_pixels = MAX_PIXELS
  pad_id = (
      processor.tokenizer.pad_token_id
      if processor.tokenizer.pad_token_id is not None
      else processor.tokenizer.eos_token_id
  )

  # --- LoRA model ---
  method = "QLoRA" if USE_QUANTIZATION else "LoRA"
  logger.info(
      "Applying %s (rank=%d, alpha=%.0f)", method, LORA_RANK, LORA_ALPHA
  )
  lora_model = get_lora_model(base_model, quantize=USE_QUANTIZATION)

  # --- Data ---
  logger.info(
      "Building DocumentVQA datasets (max_pixels=%d, max_seq_len=%d)",
      MAX_PIXELS,
      MAX_SEQ_LEN,
  )
  train_ds, eval_ds = create_datasets(
      processor, config.vision_config, BATCH_SIZE, MAX_SEQ_LEN
  )

  # --- Trainer ---
  logging_options = metrics_logger.MetricsLoggerOptions(
      log_dir="/tmp/tensorboard/qwen3_5_vl_lora",
      flush_every_n_steps=EVAL_EVERY_N_STEPS,
  )
  training_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=EVAL_EVERY_N_STEPS,
      max_steps=MAX_STEPS,
      metrics_logging_options=logging_options,
      checkpoint_root_directory=LORA_CKPT_DIR,
  )
  trainer = _Qwen3VLPeftTrainer(
      lora_model.visual,
      lora_model,
      optax.adamw(1e-3),
      training_config,
  ).with_gen_model_input_fn(make_gen_model_input_fn(pad_id))
  trainer.loss_fn = loss_fn
  trainer.eval_loss_fn = loss_fn

  logger.info("Starting %s training for %d steps", method, MAX_STEPS)
  trainer.train(train_ds, eval_ds)
  logger.info("Training complete. Checkpoints saved to %s", LORA_CKPT_DIR)


if __name__ == "__main__" and "__file__" in globals():
  # Invoke only of launched as a script and not from REPL
  main()
