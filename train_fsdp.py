"""
FSDP (Fully Sharded Data Parallel) training entry point.

Model parameters, gradients and optimizer states are sharded across GPUs.
With TinyLlama-1.1B on 2 GPUs each rank holds only ~8.8GB (half of the state),
leaving ~15GB free on a 24GB card — enough for a 4x larger batch.

Usage:
    accelerate launch --config_file configs/fsdp_config.yaml train_fsdp.py
"""

from src.train import train

# FSDP: 4x larger batch size — sharding frees up ~9GB per GPU.
# The same value OOMs under DDP; that contrast is the point of the two files.
BATCH_SIZE = 8

if __name__ == "__main__":
    # Identical to train_ddp.py apart from BATCH_SIZE — the sharding comes from
    # the accelerate config, not from this script.
    train(batch_size=BATCH_SIZE)  # save_model defaults to False
