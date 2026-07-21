# Setup Runbook — Rented GPU Box (2× 24 GB)

Copy-paste commands for going from a fresh rented instance to both training runs.
Works on any provider that gives you SSH into a multi-GPU Linux box — vast.ai,
RunPod, Lambda, a university cluster node. Everything here is plain SSH and shell;
only the renting and teardown steps go through your provider's console.

Read the STOP check in step 4 — it catches the failures that waste rental hours.

---

## 1. Set up your SSH key (one time, on your own machine)

Easiest done before renting, so the key is in place when the box comes up.

First check whether you already have a key worth reusing:

```bash
ls -la ~/.ssh/*.pub 2>/dev/null
```

Generate one dedicated to this, so it stays separate from your GitHub/work keys:

```bash
ssh-keygen -t ed25519 -C "gpu-rental" -f ~/.ssh/gpu_ed25519
```

Press Enter twice to skip the passphrase, or set one and cache it in the macOS
keychain:

```bash
ssh-add --apple-use-keychain ~/.ssh/gpu_ed25519
```

That produces two files:

| File | What it is |
|---|---|
| `~/.ssh/gpu_ed25519` | **Private key — secret.** Never share, never paste anywhere. |
| `~/.ssh/gpu_ed25519.pub` | Public key. This is the one you hand to the provider. |

Fix permissions (SSH silently refuses keys that are too readable):

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/gpu_ed25519
chmod 644 ~/.ssh/gpu_ed25519.pub
```

Copy the **public** key to your clipboard:

```bash
# macOS
pbcopy < ~/.ssh/gpu_ed25519.pub

# Linux
xclip -selection clipboard < ~/.ssh/gpu_ed25519.pub

# or just print it and copy by hand
cat ~/.ssh/gpu_ed25519.pub
```

> **Only ever paste the `.pub` file.** It starts with `ssh-ed25519 AAAA…`. If what
> you are looking at starts with `-----BEGIN OPENSSH PRIVATE KEY-----`, that is the
> private key and it must never leave your machine.

Paste it into your provider's SSH-keys page.

> **Where providers differ:** some hold keys per-account, some per-instance, some
> both. vast.ai does both — an account key is injected into instances created
> *afterwards*, while its per-instance "Manage SSH Keys" dialog attaches a key to an
> already-running box and takes effect immediately. If you rented before adding a
> key, look for the per-instance option before assuming you have to rebuild.

### Connecting with this key

Because the key lives in `gpu_ed25519` rather than the default `id_ed25519`, SSH
will **not** find it on its own. Every connection needs `-i` pointing at the private
key (the file with no `.pub`):

```bash
ssh -i ~/.ssh/gpu_ed25519 -p <PORT> root@<HOST>
```

Same flag for copying files off the box later — note `scp` wants a capital `-P` for
the port where `ssh` wants lowercase:

```bash
scp -i ~/.ssh/gpu_ed25519 -P <PORT> root@<HOST>:/path/to/file .
```

> **This is the step people trip on.** The connect command your provider shows you
> is usually just `ssh -p 12345 root@some.host` with no `-i`. Pasted as is, it fails
> with `Permission denied (publickey)` — not because your key is wrong, but because
> SSH never offered it.

Save yourself the flag by adding a shortcut to `~/.ssh/config`, filling in the host
and port once the instance is running:

```
Host gpubox
    HostName <HOST>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/gpu_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```

Then plain `ssh gpubox` works. `IdentitiesOnly yes` matters if you have several keys
— without it SSH offers them all and the server can cut you off with "Too many
authentication failures" before reaching the right one.

## 2. Rent the instance

Book **2 GPUs on one machine**, 24 GB each, and — critically — **enough disk**.
Providers commonly default to ~10 GB, which is not enough: torch wheels alone are
~3 GB, and the model plus HF cache adds another ~3 GB.

| Setting | Value | Why |
|---|---|---|
| GPU | 2× RTX 3090 (or 4090 / A5000) | 24 GB each, Ampere so bf16 works |
| Disk | **50 GB** | deps ~6 GB + model ~2.5 GB + cache; 10 GB will fail mid-`uv sync` |
| CUDA | ≥ 12.1 | host driver must support the torch wheel |
| Image | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | any CUDA base is fine; uv installs its own torch |

Two GPUs must be on the **same physical machine** — this is single-node distributed
training, not multi-node. Anything sold as "2 GPUs" in one instance qualifies.

## 3. Connect

Take the host and port from your provider's console, and add `-i` with your key:

```bash
ssh -i ~/.ssh/gpu_ed25519 -p <PORT> root@<HOST>
```

Or just `ssh gpubox` if you added the config block in step 1.

Landing at a `root@...:~#` prompt means the key is working. If it asks for a password
or refuses you, jump to `Permission denied (publickey)` in troubleshooting.

## 4. STOP — verify the hardware before installing anything

Fail fast here rather than 10 minutes into a `uv sync`.

```bash
# Two GPUs, 24GB each, compute capability 8.6
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# Interconnect between the two cards
nvidia-smi topo -m

# Shared memory — Docker defaults to 64MB, which crashes dataloader workers
df -h /dev/shm

# Disk actually provisioned
df -h /
```

**What you need to see:**

- **Two rows** from the first command. One row means you got a single-GPU box — destroy it, this demo needs two.
- **`compute_cap` of 8.0 or higher.** 7.5 is Turing and has no bf16; the configs will fail.
- **`/dev/shm` of at least 1 GB.** If it shows 64M, add `dataloader_num_workers=0` to `SFTConfig` in `src/train.py`.
- **At least ~40 GB free on `/`.** Less than that and `uv sync` will die partway.

## 5. Install uv, clone, sync

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/shivam13juna/Accelerate-LLM-FineTune.git
cd Accelerate-LLM-FineTune

uv sync
```

Confirm torch actually sees both GPUs, and that bf16 is genuinely supported:

```bash
uv run python -c "
import torch
print('torch      ', torch.__version__)
print('cuda avail ', torch.cuda.is_available())
print('gpu count  ', torch.cuda.device_count())
print('bf16 ok    ', torch.cuda.is_bf16_supported())
"
```

`gpu count 2` and `bf16 ok True` are both required. Anything else, stop and fix it here.

## 6. Pre-fetch the model and dataset

Do this once, outside the training run — otherwise both ranks race to download the
same files on first launch and the timings are polluted.

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('model cached')
"

uv run python -c "
from datasets import load_dataset
load_dataset('yahma/alpaca-cleaned', split='train')
print('dataset cached')
"
```

## 7. Run inside tmux

The runs outlive your SSH session this way — a dropped connection otherwise kills
training and wastes the rental.

```bash
tmux new -s train
# detach: Ctrl-b then d
# reattach: tmux attach -t train
```

## 8. Smoke test first (~1 minute)

Shrink the dataset so you find out whether the whole pipeline works before
committing to a full run.

```bash
sed -i 's/^NUM_SAMPLES = 10_000/NUM_SAMPLES = 200/' src/train.py

uv run accelerate launch --config_file configs/fsdp_config.yaml train_fsdp.py
```

Restore when it passes:

```bash
sed -i 's/^NUM_SAMPLES = 200/NUM_SAMPLES = 10_000/' src/train.py
```

## 9. The two runs

```bash
# DDP — batch 2 per GPU, effective 16
uv run accelerate launch --config_file configs/ddp_config.yaml train_ddp.py

# FSDP — batch 8 per GPU, effective 64
uv run accelerate launch --config_file configs/fsdp_config.yaml train_fsdp.py
```

Each prints its peak memory per rank at the end. That comparison is the demo:

```
[rank 0] peak GPU memory: ~20 GB / 24.0 GB     # DDP,  batch 2
[rank 0] peak GPU memory: ~17 GB / 24.0 GB     # FSDP, batch 8
```

FSDP uses less memory while doing 4× more work per step.

## 10. The OOM demo

```bash
sed -i 's/^BATCH_SIZE = 2/BATCH_SIZE = 8/' train_ddp.py

uv run accelerate launch --config_file configs/ddp_config.yaml train_ddp.py
# expected: torch.OutOfMemoryError: CUDA out of memory

sed -i 's/^BATCH_SIZE = 8/BATCH_SIZE = 2/' train_ddp.py
```

DDP at batch 8 needs ~26 GB against a 24 GB card. FSDP at the same batch size needs
~17 GB, because sharding halves the 17.6 GB fixed cost.

> Run this once before class. The activation estimate is version-dependent, so
> confirm it actually OOMs on your specific instance rather than finding out live.

## 11. Watch memory in a second terminal

```bash
watch -n 1 nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

Under DDP both GPUs sit at nearly identical high memory. Under FSDP both drop by
roughly half — visible in real time, which makes the point better than the logs do.

---

## Troubleshooting

### `Permission denied (publickey)`

Two causes, and the first is far more common.

**1. You forgot `-i`.** The connect command from your provider's console has no `-i`,
so SSH never offers your `gpu_ed25519` key at all — it only tries default names like
`id_ed25519` and `id_rsa`. Nothing is wrong with the key; it was simply not presented:

```bash
ssh -i ~/.ssh/gpu_ed25519 -p <PORT> root@<HOST>
```

Confirm which keys SSH actually offered:

```bash
ssh -v -i ~/.ssh/gpu_ed25519 -p <PORT> root@<HOST> 2>&1 | grep -i 'offering\|publickey'
```

**2. The key really is not on the box.** Common when it was added to your account
*after* this instance started — account-level keys usually only propagate to
instances created afterwards. Check your provider for a per-instance key option,
which typically applies without a restart.

### `REMOTE HOST IDENTIFICATION HAS CHANGED`

Not an attack — providers recycle IPs and ports between instances, so a host you
trusted last week is now different hardware. Drop the stale entry:

```bash
ssh-keygen -R "[<HOST>]:<PORT>"
```

### `Too many authentication failures`

SSH is offering every key you own and the server cuts the connection first. Force
just the one:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/gpu_ed25519 -p <PORT> root@<HOST>
```

### Training hangs at 0% with no error — the classic 3090 failure

NVIDIA disables peer-to-peer over PCIe on GeForce cards, so NCCL can hang forever
during init on consumer GPUs. This is the single most common multi-GPU problem on
rented consumer hardware. Fix:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

Then re-launch. To confirm that was the cause before applying the workaround:

```bash
NCCL_DEBUG=INFO uv run accelerate launch --config_file configs/ddp_config.yaml train_ddp.py
```

Look for NCCL stalling after the topology detection lines.

### `uv sync` fails partway with a disk error

```bash
df -h /
```

You provisioned too little disk. Most providers cannot resize a running instance —
destroy it and recreate with 50 GB.

### Dataloader workers crash / bus error

`/dev/shm` is too small. Add `dataloader_num_workers=0` to `SFTConfig` in
`src/train.py`.

### `ValueError` about mixed precision mismatch

The accelerate config and `SFTConfig` disagree. Both must say bf16 — check
`mixed_precision: bf16` in the YAML against `bf16=True` in `src/train.py`.

### FSDP OOMs too

Less headroom on your instance than estimated. Add to `SFTConfig`:

```python
gradient_checkpointing=True
```

Costs ~30% speed, saves a large amount of activation memory.

---

## Teardown

Copy anything you want to keep off the box first — instance storage does not survive:

```bash
scp -i ~/.ssh/gpu_ed25519 -P <PORT> root@<HOST>:~/Accelerate-LLM-FineTune/output.log .
```

Then destroy the instance from your provider's console.

> **Stopping is usually not enough.** Most GPU rental providers keep billing for
> storage on a stopped instance — only destroying it stops charges entirely. Check
> your provider's rules rather than assuming a stopped box is a free box.
