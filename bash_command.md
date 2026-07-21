# Setup Runbook — vast.ai (2× RTX 3090)

Copy-paste commands for going from a fresh rented instance to both training runs.
Read the STOP check in step 4 — it catches the failures that waste rental hours.

---

## 1. Set up your SSH key (one time, on your own machine)

Easiest done **before** renting, though it is not a hard requirement — vast.ai keeps
SSH keys at two levels:

| Level | Where | Behaviour |
|---|---|---|
| **Account** — "Your SSH Keys" | Console → Keys | Injected into every **new** instance at start. Set once, forget. |
| **Instance** — "Instance SSH Keys" | Instance row → Manage SSH Keys | Attached to one **running** instance, applies immediately. No restart. |

So doing it first means every future instance just works. If you have already rented
and cannot get in, you are not stuck — attach a key to the running instance via the
Manage SSH Keys dialog (see troubleshooting below).

First check whether you already have a key worth reusing:

```bash
ls -la ~/.ssh/*.pub 2>/dev/null
```

Generate one dedicated to vast.ai, so it stays separate from your GitHub/work keys:

```bash
ssh-keygen -t ed25519 -C "vast.ai" -f ~/.ssh/vast_ed25519
```

Press Enter twice to skip the passphrase, or set one and cache it in the macOS
keychain:

```bash
ssh-add --apple-use-keychain ~/.ssh/vast_ed25519
```

That produces two files:

| File | What it is |
|---|---|
| `~/.ssh/vast_ed25519` | **Private key — secret.** Never share, never paste anywhere. |
| `~/.ssh/vast_ed25519.pub` | Public key. This is the one you give to vast.ai. |

Fix permissions (SSH silently refuses keys that are too readable):

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/vast_ed25519
chmod 644 ~/.ssh/vast_ed25519.pub
```

Copy the **public** key to your clipboard:

```bash
# macOS
pbcopy < ~/.ssh/vast_ed25519.pub

# Linux
xclip -selection clipboard < ~/.ssh/vast_ed25519.pub

# or just print it and copy by hand
cat ~/.ssh/vast_ed25519.pub
```

> **Only ever paste the `.pub` file.** It starts with `ssh-ed25519 AAAA…`. If what
> you are looking at starts with `-----BEGIN OPENSSH PRIVATE KEY-----`, that is the
> private key and it must never leave your machine.

Add it to your account in the vast.ai console under **Keys → SSH Keys → New**, paste,
save. Or via the CLI:

```bash
pip install --upgrade vastai
vastai set api-key <YOUR_API_KEY>

vastai create ssh-key "$(cat ~/.ssh/vast_ed25519.pub)"
vastai show ssh-keys
```

Optionally add a shortcut to `~/.ssh/config` so you can type `ssh vast` instead of
the full command — fill in HostName and Port once the instance is running:

```
Host vast
    HostName <HOST>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/vast_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```

`IdentitiesOnly yes` matters if you have several keys — without it SSH offers them
all and the server can cut you off with "Too many authentication failures" before
reaching the right one.

## 2. Rent the instance

Filter for **2 GPUs on one machine**, 24 GB each, and — critically — **enough disk**.
vast.ai defaults to ~10 GB, which is not enough: torch wheels alone are ~3 GB, and
the model plus HF cache adds another ~3 GB.

| Setting | Value | Why |
|---|---|---|
| GPU | 2× RTX 3090 (or 4090 / A5000) | 24 GB each, Ampere so bf16 works |
| Disk | **50 GB** | deps ~6 GB + model ~2.5 GB + cache; 10 GB will fail mid-`uv sync` |
| CUDA | ≥ 12.1 | host driver must support the torch wheel |
| Image | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | any CUDA base is fine; uv installs its own torch |

Via the web console is most reliable. If you prefer the CLI:

```bash
# find offers
vastai search offers 'num_gpus=2 gpu_name=RTX_3090 disk_space>=50 cuda_vers>=12.1' -o 'dph+'

# create (flags drift between vastai versions — check `vastai create instance --help`)
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  --disk 50 --ssh --direct

vastai show instances
```

## 3. Connect

Grab the host and port from the console's Connect button, or `vastai ssh-url <ID>`.

```bash
ssh -i ~/.ssh/vast_ed25519 -p <PORT> root@<HOST>
```

Or just `ssh vast` if you added the config block above.

vast.ai gives you two connection paths. **Direct** (`--direct` at create time) connects
straight to the machine's IP and is faster. **Proxy** routes through
`root@sshN.vast.ai` and works when the host has no open inbound ports. Either is
fine here; direct is nicer for file copies.

## 4. STOP — verify the hardware before installing anything

Fail fast here rather than 10 minutes into a `uv sync`.

```bash
# Two GPUs, 24GB each, compute capability 8.6
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

# Interconnect between the two cards
nvidia-smi topo -m

# Shared memory — Docker defaults to 64MB, which crashes dataloader workers
df -h /dev/shm
```

**What you need to see:**

- **Two rows** from the first command. One row means you got a single-GPU box — destroy it, this demo needs two.
- **`compute_cap` of 8.0 or higher.** 7.5 is Turing and has no bf16; the configs will fail.
- **`/dev/shm` of at least 1 GB.** If it shows 64M, add `dataloader_num_workers=0` to `SFTConfig` in `src/train.py`, or recreate the instance with a larger `--shm-size`.

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

Usually the key was added to your vast.ai account *after* this instance started, so it
was never injected — account keys only propagate to instances created afterwards.

**No restart needed.** Attach the key straight to the running instance: find it in the
console, open **Manage SSH Keys**, and paste your public key. The dialog shows
"Your SSH Keys" (account-level) beside "Instance SSH Keys" (this box only); the second
list is the one that has to contain your key. It applies immediately.

The CLI equivalent, if the console is awkward:

```bash
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/vast_ed25519.pub)"
vastai show ssh-keys
```

To see which key SSH is actually offering:

```bash
ssh -v -i ~/.ssh/vast_ed25519 -p <PORT> root@<HOST> 2>&1 | grep -i 'offering\|publickey'
```

### `REMOTE HOST IDENTIFICATION HAS CHANGED`

Not an attack — vast.ai recycles IPs and ports between instances, so a host you
trusted last week is now different hardware. Drop the stale entry:

```bash
ssh-keygen -R "[<HOST>]:<PORT>"
```

### `Too many authentication failures`

SSH is offering every key you own and the server cuts the connection first. Force
just the one:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/vast_ed25519 -p <PORT> root@<HOST>
```

### Training hangs at 0% with no error — the classic 3090 failure

NVIDIA disables peer-to-peer over PCIe on GeForce cards, so NCCL can hang forever
during init on consumer GPUs. This is the single most common vast.ai multi-GPU
problem. Fix:

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

You provisioned too little disk. Disk cannot be resized on a running vast.ai
instance — destroy it and recreate with `--disk 50`.

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

**Billing continues while the instance is stopped.** Only destroying it stops charges.

Copy anything you want to keep off the box first:

```bash
scp -i ~/.ssh/vast_ed25519 -P <PORT> root@<HOST>:~/Accelerate-LLM-FineTune/output.log .
```

Then destroy it:

```bash
vastai destroy instance <INSTANCE_ID>
```

Or use the Destroy button in the console. Instance storage is gone for good.
