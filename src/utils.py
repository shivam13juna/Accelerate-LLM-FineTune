"""GPU memory reporting utilities.

These make the DDP vs FSDP difference *visible* rather than asserted — run both
strategies and compare the peak numbers each rank prints at the end.
"""

import torch


def reset_peak_memory():
    """Zero the peak-memory counter so the reading reflects training only."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def report_peak_memory(rank):
    """Print this rank's peak GPU memory.

    Under DDP every rank holds a full model + gradients + optimizer, so all
    ranks report roughly the same large number. Under FSDP each rank holds
    only 1/N of that state, so the peak should drop noticeably.

    Args:
        rank: This process's index (0 to world_size-1). Used only to label the
            output, since all ranks print to the same terminal.
    """
    if not torch.cuda.is_available():
        return

    # torch's own high-water mark, in bytes → GB. Reads lower than nvidia-smi:
    # it excludes the CUDA context (~0.5GB) and blocks the caching allocator is
    # holding but not using.
    peak = torch.cuda.max_memory_allocated() / 1024**3
    # Device 0 is correct on every rank — accelerate sets CUDA_VISIBLE_DEVICES
    # per process, so each one sees its assigned GPU as its only device.
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[rank {rank}] peak GPU memory: {peak:.1f} GB / {total:.1f} GB")
