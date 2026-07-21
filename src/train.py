"""Core training logic using HuggingFace SFTTrainer.

Both entry points (train_ddp.py, train_fsdp.py) call `train()` below. Nothing in
this file knows which distributed strategy is running — that is decided entirely
by the accelerate YAML passed on the command line.
"""

import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from src.data import load_tokenizer, get_dataset
from src.utils import report_peak_memory, reset_peak_memory

# ── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params → ~17.6GB fixed cost
DATASET_NAME = "yahma/alpaca-cleaned"  # 51k instruction/response pairs (deduped Alpaca)
MAX_SEQ_LENGTH = 512      # tokens per example; the main driver of activation memory
DEFAULT_BATCH_SIZE = 2    # per device (conservative default for DDP)
GRAD_ACCUM_STEPS = 4      # forward/backward passes before one optimizer update
NUM_EPOCHS = 1            # one pass is enough to observe the memory behaviour
LR = 2e-5                 # typical full-finetune range; LoRA would use ~10x higher
SEED = 42                 # identical weight init and data order on every rank
LOG_EVERY = 50            # optimizer steps between loss prints
NUM_SAMPLES = 10_000      # subset of the dataset; drop to ~200 for a smoke test
OUTPUT_DIR = "output"     # checkpoints, logs, and the final model if saved


def train(batch_size=DEFAULT_BATCH_SIZE, save_model=False):
    """Run one supervised finetune.

    Args:
        batch_size: Examples per GPU per forward pass. This is the only knob the
            DDP and FSDP entry points differ on (2 vs 8), and the whole reason
            the two strategies behave differently on identical hardware.
        save_model: Write model + tokenizer to OUTPUT_DIR when training ends.
            Off by default — a full TinyLlama checkpoint is ~4.4GB, and the
            memory comparison does not need the weights kept.
    """
    tokenizer = load_tokenizer(MODEL_NAME)

    # Load in fp32 on purpose. Together with bf16=True below this gives standard
    # mixed precision: fp32 master weights + fp32 optimizer states, with bf16 used
    # only for the forward/backward compute. Loading in bf16 instead would make
    # AdamW keep bf16 optimizer states too — cheaper, but numerically worse, and
    # it would halve the memory that this DDP-vs-FSDP demo is built to expose.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,              # repo id on the HF Hub, or a local path
        dtype=torch.float32,     # load-bearing — see comment above
    )

    # `tokenizer` and MAX_SEQ_LENGTH are accepted but unused: SFTTrainer does the
    # tokenizing, and truncation comes from max_length in SFTConfig below. This
    # call just returns rows with a single "text" column of formatted strings.
    dataset = get_dataset(DATASET_NAME, tokenizer, MAX_SEQ_LENGTH, NUM_SAMPLES)

    # SFTConfig subclasses transformers.TrainingArguments, so most fields below
    # are inherited — only max_length is TRL's own.
    sft_config = SFTConfig(
        # Where checkpoints, logs and the final model land. Required even with
        # save_strategy="no", because Trainer still writes its own state there.
        output_dir=OUTPUT_DIR,

        # PER GPU, not total. With 2 GPUs, batch_size=2 puts 4 examples in flight
        # per step. This is the number that OOMs DDP at 8 but not FSDP.
        per_device_train_batch_size=batch_size,

        # Accumulate gradients over this many passes before stepping the
        # optimizer. Multiplies the effective batch at almost no memory cost —
        # gradients land in a buffer that already exists.
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        num_train_epochs=NUM_EPOCHS,  # one pass over NUM_SAMPLES
        learning_rate=LR,             # constant-ish; no scheduler configured here

        # bf16 autocast for forward/backward compute only — weights and optimizer
        # states stay fp32 (see from_pretrained above). This MUST match
        # `mixed_precision` in the accelerate YAML or Trainer raises at startup.
        bf16=True,

        logging_steps=LOG_EVERY,  # print loss every N optimizer steps

        # No intermediate checkpoints. Each is ~4.4GB, rented disk is small, and
        # the final save is handled by save_model below instead.
        save_strategy="no",

        seed=SEED,  # seeds weight init, shuffling, and dropout on every rank

        # Truncate examples to this many tokens. TRL's own field — renamed from
        # max_seq_length in trl 0.20.
        max_length=MAX_SEQ_LENGTH,
    )

    # world_size is however many processes accelerate actually launched, so this
    # stays correct whether you run on 1 GPU or 8.
    effective_batch = batch_size * GRAD_ACCUM_STEPS * sft_config.world_size
    # process_index is this rank's id; 0 is the main process. Guarding the prints
    # keeps the header from appearing once per GPU.
    if sft_config.process_index == 0:
        print(f"GPUs (world size):     {sft_config.world_size}")
        print(f"Per-device batch size: {batch_size}")
        print(f"Effective batch size:  {effective_batch}")

    trainer = SFTTrainer(
        model=model,            # the fp32 model loaded above
        args=sft_config,        # everything configured in SFTConfig
        train_dataset=dataset,  # needs a "text" column; SFTTrainer tokenizes it

        # Formerly `tokenizer=`. Renamed because newer models need image/audio
        # processors here too. Used to tokenize the dataset, and saved alongside
        # the weights so the checkpoint is self-contained.
        processing_class=tokenizer,
    )

    reset_peak_memory()  # zero the counter so the reading covers training only
    trainer.train()
    report_peak_memory(sft_config.process_index)  # each rank prints its own peak

    if save_model:
        trainer.save_model(OUTPUT_DIR)       # weights + config
        tokenizer.save_pretrained(OUTPUT_DIR)  # tokenizer files, saved separately
        print(f"Model saved to {OUTPUT_DIR}/")
    else:
        print("Skipping model save (save_model=False)")


if __name__ == "__main__":
    # Only hit when running this file directly (single GPU, no accelerate).
    # The DDP/FSDP entry points call train() themselves with their own batch size.
    train(save_model=True)
