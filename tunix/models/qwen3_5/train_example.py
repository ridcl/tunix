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
  - Data processing is split into two stages:
      1. _PrepareConversation: dataset-specific, converts raw HF items into
         OpenAI-format conversation dicts (PIL images kept as-is).
      2. _BatchingLoader: common, collects B conversations and calls
         encode_messages() to produce a single EncodedBatch.
  - encode_messages() handles tokenisation, loss masking, 3-D M-RoPE position
    computation, and VisionGridData precomputation — all in the data pipeline,
    outside the JAX JIT boundary.
  - EncodedBatch.pixel_values packs all patches as [P, C]; no per-item padding.

Usage::

    python -m tunix.models.qwen3_5.train_example
"""

from __future__ import annotations

import logging
import os
from typing import Any

import datasets
from grain import python as grain
import huggingface_hub
import jax
import jax.numpy as jnp
import optax
import qwix
from transformers import AutoProcessor
from tunix.models.qwen3_5 import model as model_lib
from tunix.models.qwen3_5 import params as params_lib
from tunix.models.qwen3_5.utils import encode_messages
from tunix.models.qwen3_5.utils import EncodedBatch
from tunix.models.qwen3vl.vision import VisionGridData
from tunix.rl import reshard as reshard_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft.utils import show_hbm_usage

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(
    logging.INFO
)  # override if basicConfig was a no-op
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-4B"

BATCH_SIZE = 1
MAX_SEQ_LEN = 2048
# Vision encoder does full self-attention over all patches before spatial
# merging.  At patch_size=16, a 2000×2000 image produces ~15K patches and
# the float32 attention-score matrix [n_heads, n, n] hits ~14.6 GiB.
# Cap images so n_patches stays within the available pool.
# 1280 px → 80×80 = 6400 patches → [16, 6400, 6400] float32 ≈ 2.6 GiB.
MAX_IMAGE_SIZE = 1280

USE_QUANTIZATION = False  # True -> QLoRA, False -> LoRA
LORA_RANK = 16
LORA_ALPHA = float(2 * LORA_RANK)

MAX_STEPS = 100
EVAL_EVERY_N_STEPS = 20

LORA_CKPT_DIR = "/tmp/qwen3_5_vl_lora_ckpts"

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
# Dataset
# ---------------------------------------------------------------------------


class _PrepareConversation(grain.MapTransform):
  """Convert one DocumentVQA example to an OpenAI-format conversation dict.

  This step is dataset-specific.  It returns a plain Python dict with a
  single key ``conversation`` holding a list of message dicts.  PIL images
  are kept as-is; no tokenisation happens here.
  """

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    image = element["image"].convert("RGB")
    if max(image.size) > MAX_IMAGE_SIZE:
      image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
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
    return {"conversation": conversation}


class _BatchingLoader:
  """Collect conversations from a grain DataLoader and encode them in batches.

  This step is common across datasets.  It wraps a grain DataLoader that
  yields individual ``{"conversation": ...}`` dicts and groups them into
  batches of ``batch_size``, then calls ``encode_messages()`` once per batch
  to tokenise, build the loss mask, and compute M-RoPE positions.

  Incomplete trailing batches are dropped.
  """

  def __init__(
      self,
      loader: grain.DataLoader,
      batch_size: int,
      processor: AutoProcessor,
      vcfg: model_lib.VisionModelConfig,
      max_seq_len: int,
  ):
    self._loader = loader
    self._batch_size = batch_size
    self._processor = processor
    self._vcfg = vcfg
    self._max_seq_len = max_seq_len

  def __iter__(self):
    buffer: list[list[dict[str, Any]]] = []
    for item in self._loader:
      buffer.append(item["conversation"])
      if len(buffer) == self._batch_size:
        yield encode_messages(
            self._processor,
            buffer,
            loss_roles={"assistant"},
            max_seq_len=self._max_seq_len,
            vcfg=self._vcfg,
            padding="max_length",
            truncation=True,
        )
        buffer = []


def create_datasets(
    processor: AutoProcessor,
    vcfg: model_lib.VisionModelConfig,
    batch_size: int,
    max_seq_len: int,
) -> tuple[_BatchingLoader, _BatchingLoader]:
  """Return (train_loader, eval_loader) for HuggingFaceM4/DocumentVQA."""
  hf_ds = datasets.load_dataset("HuggingFaceM4/DocumentVQA")
  train_hf = hf_ds["train"]
  eval_hf = hf_ds["validation"]

  ops = [_PrepareConversation()]

  def _make_loader(hf_split, num_epochs):
    grain_loader = grain.DataLoader(
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
    return _BatchingLoader(
        grain_loader, batch_size, processor, vcfg, max_seq_len
    )

  return _make_loader(train_hf, num_epochs=3), _make_loader(
      eval_hf, num_epochs=1
  )


# ---------------------------------------------------------------------------
# Model input conversion and loss
# ---------------------------------------------------------------------------


def _gen_model_input_fn(batch: EncodedBatch) -> dict:
  """Convert an EncodedBatch to model kwargs."""
  return {
      "input_tokens": jnp.array(batch.input_tokens),
      "input_mask": jnp.array(batch.input_mask).astype(jnp.bool_),
      "completion_mask": jnp.array(batch.completion_mask),
      "positions": jnp.array(batch.positions),
      "pixel_values": jnp.array(batch.pixel_values, dtype=jnp.bfloat16),
      "vision_grid": batch.vision_grid,
  }


def loss_fn(
    model: model_lib.Qwen3_5,
    input_tokens: jax.Array,  # [B, L]
    input_mask: jax.Array,  # [B, L] — padding mask (1=real, 0=pad)
    completion_mask: jax.Array,  # [B, L] — 1 for tokens to include in loss
    positions: jax.Array,  # [3, B, L]
    pixel_values: jax.Array,  # [B*n_patches, C]
    vision_grid: VisionGridData,  # pre-computed outside JIT
) -> jax.Array:
  """Cross-entropy loss over answer tokens only."""
  logits, _ = model(
      input_tokens,
      positions=positions,
      pixel_values=pixel_values,
      vision_grid=vision_grid,
      cache=None,
      input_mask=input_mask,
  )
  # Shift by 1: predict token t+1 from the hidden state at position t.
  # Keep logits in bfloat16 to avoid materialising a float32 [B, L, V] tensor
  # (~2.4 GiB/GPU unsharded), which exhausts the memory pool during
  # cross-entropy computation.  Cast only the small [B, L-1] per-token loss.
  logits = logits[:, :-1, :]  # [B, L-1, V] bfloat16
  targets = input_tokens[:, 1:]  # [B, L-1]
  mask = completion_mask[:, 1:].astype(jnp.float32)  # [B, L-1]

  token_loss = optax.softmax_cross_entropy_with_integer_labels(
      logits, targets
  ).astype(
      jnp.float32
  )  # [B, L-1]
  return jnp.sum(token_loss * mask) / jnp.sum(mask)


# ---------------------------------------------------------------------------
# LoRA / QLoRA helper
# ---------------------------------------------------------------------------

_LORA_TARGETS = ".*q_proj|.*k_proj|.*gate_proj|.*up_proj|.*down_proj"
# _LORA_TARGETS = ".*q_proj|.*k_proj"


def get_lora_model(
    base_model: model_lib.Qwen3_5,
    mesh: jax.sharding.Mesh | None = None,
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
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )
  # TODO: do we still need it?
  # Reshard LoRA params onto the mesh so XLA's SPMD partitioner gets explicit
  # sharding for every parameter.  Without this, QWIX-created LoRA params lack
  # NNX out_sharding metadata, leaving their placement ambiguous and causing
  # extremely long XLA compilation with TP sharding.
  if mesh is not None:
    lora_model = reshard_lib.reshard_model_to_mesh(lora_model, mesh)
  return lora_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  jax.config.update("jax_explain_cache_misses", True)  # Show compilation
  jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
  os.makedirs(LORA_CKPT_DIR, exist_ok=True)

  # --- Load model ---
  config = model_lib.ModelConfig.qwen3_5_4b_vl()
  config.remat_config = model_lib.RematConfig.BLOCK
  model_dir = resolve_model_dir(MODEL_ID)

  logger.info("Loading model from %s", model_dir)
  mesh = jax.make_mesh((1, 2), ("fsdp", "tp"))
  base_model = params_lib.create_model_from_safe_tensors(
      model_dir, config, mesh=mesh, dtype=jnp.bfloat16
  )
  show_hbm_usage()

  # --- Processor (tokenizer + image processor) ---
  processor = AutoProcessor.from_pretrained(model_dir)

  # --- LoRA model ---
  method = "QLoRA" if USE_QUANTIZATION else "LoRA"
  logger.info(
      "Applying %s (rank=%d, alpha=%.0f)", method, LORA_RANK, LORA_ALPHA
  )
  lora_model = get_lora_model(base_model, mesh=mesh, quantize=USE_QUANTIZATION)
  show_hbm_usage()

  # --- Data ---
  logger.info("Building DocumentVQA datasets (max_seq_len=%d)", MAX_SEQ_LEN)
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
  trainer = peft_trainer.PeftTrainer(
      lora_model,
      optax.adamw(1e-3),
      training_config,
  ).with_gen_model_input_fn(_gen_model_input_fn)
  trainer.loss_fn = loss_fn
  trainer.eval_loss_fn = loss_fn

  logger.info("Starting %s training for %d steps", method, MAX_STEPS)
  with mesh:
    trainer.train(train_ds, eval_ds=None)
    # trainer.train(train_ds, eval_ds)
  logger.info("Training complete. Checkpoints saved to %s", LORA_CKPT_DIR)


if __name__ == "__main__" and "__file__" in globals():
  # Invoke only of launched as a script and not from REPL
  main()
