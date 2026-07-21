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

# DDP: conservative batch size — full model replica on every GPU
BATCH_SIZE = 2

if __name__ == "__main__":
    train(batch_size=BATCH_SIZE)
