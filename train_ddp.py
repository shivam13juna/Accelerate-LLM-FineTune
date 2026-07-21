"""
DDP (Distributed Data Parallel) training entry point.

Each GPU holds a full copy of the model, its gradients and the optimizer state.
With TinyLlama-1.1B that is ~17.6GB per GPU before a single activation is
allocated, leaving only ~6GB on a 24GB card — which is what limits the
per-device batch size to 2.

Usage:
    accelerate launch --config_file configs/ddp_config.yaml train_ddp.py
"""

from src.train import train

# DDP: conservative batch size — full model replica on every GPU.
# Raise this to 8 to reproduce the out-of-memory error on a 24GB card.
BATCH_SIZE = 2

if __name__ == "__main__":
    # Every launched process runs this file top to bottom, so all ranks call
    # train() with the same batch size. Per-rank differences come from
    # accelerate's environment, not from anything here.
    train(batch_size=BATCH_SIZE)  # save_model defaults to False
