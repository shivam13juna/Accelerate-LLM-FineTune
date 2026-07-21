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
    """
    if not torch.cuda.is_available():
        return

    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[rank {rank}] peak GPU memory: {peak:.1f} GB / {total:.1f} GB")
