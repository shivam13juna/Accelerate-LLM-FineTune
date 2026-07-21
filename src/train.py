"""Core training logic using HuggingFace SFTTrainer."""

import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from src.data import load_tokenizer, get_dataset
from src.utils import report_peak_memory, reset_peak_memory

# ── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 2   # per device (conservative default for DDP)
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 1
LR = 2e-5
SEED = 42
LOG_EVERY = 50
NUM_SAMPLES = 10_000
OUTPUT_DIR = "output"


def train(batch_size=DEFAULT_BATCH_SIZE, save_model=False):
    tokenizer = load_tokenizer(MODEL_NAME)

    # Load in fp32 on purpose. Together with bf16=True below this gives standard
    # mixed precision: fp32 master weights + fp32 optimizer states, with bf16 used
    # only for the forward/backward compute. Loading in bf16 instead would make
    # AdamW keep bf16 optimizer states too — cheaper, but numerically worse, and
    # it would halve the memory that this DDP-vs-FSDP demo is built to expose.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)

    dataset = get_dataset(DATASET_NAME, tokenizer, MAX_SEQ_LENGTH, NUM_SAMPLES)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=LOG_EVERY,
        save_strategy="no",
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
    )

    # world_size is however many processes accelerate actually launched, so this
    # stays correct whether you run on 1 GPU or 8.
    effective_batch = batch_size * GRAD_ACCUM_STEPS * sft_config.world_size
    if sft_config.process_index == 0:
        print(f"GPUs (world size):     {sft_config.world_size}")
        print(f"Per-device batch size: {batch_size}")
        print(f"Effective batch size:  {effective_batch}")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    reset_peak_memory()
    trainer.train()
    report_peak_memory(sft_config.process_index)

    if save_model:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}/")
    else:
        print("Skipping model save (save_model=False)")


if __name__ == "__main__":
    train(save_model=True)
