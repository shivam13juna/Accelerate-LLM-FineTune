# LLM Finetuning: DDP vs FSDP with SFTTrainer + Accelerate

Finetune **TinyLlama-1.1B** on the **Alpaca** dataset using HuggingFace SFTTrainer.
Same training logic, two distributed strategies — just swap the accelerate config file.

This project demonstrates a key advantage of FSDP over DDP: **FSDP can train with much larger batch sizes** because it shards model state across GPUs, freeing per-GPU memory for activations.

## Prerequisites

- 2× NVIDIA GPUs with **24 GB** each (tested on RTX 3090)
- **Ampere or newer** (compute capability 8.0+) — bf16 is not supported on Turing or older, so a 2080/2060 will not run these configs as written
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

```bash
# Install dependencies
uv sync
```

## Understanding DDP vs FSDP

### The Core Difference

Both DDP and FSDP are strategies for training across multiple GPUs, but they manage GPU memory very differently.

**DDP (Distributed Data Parallel)** — each GPU holds a **full copy** of everything:

```
GPU 0: [Full Model 4.4GB] + [Full Grads 4.4GB] + [Full Optimizer 8.8GB] + [Activations ??]
GPU 1: [Full Model 4.4GB] + [Full Grads 4.4GB] + [Full Optimizer 8.8GB] + [Activations ??]
```

**FSDP (Fully Sharded Data Parallel)** — model, gradients and optimizer are **sharded** (split) across GPUs:

```
GPU 0: [1/2 Model 2.2GB] + [1/2 Grads 2.2GB] + [1/2 Optimizer 4.4GB] + [Activations !!!!!!]
GPU 1: [1/2 Model 2.2GB] + [1/2 Grads 2.2GB] + [1/2 Optimizer 4.4GB] + [Activations !!!!!!]
```

### The 16 Bytes Per Parameter Rule

Full finetuning with Adam costs about **16 bytes per parameter**, and this holds
whether or not you use mixed precision — because mixed precision still keeps
fp32 master weights and fp32 optimizer states:

| Component | Bytes/param | TinyLlama-1.1B |
|---|---|---|
| Model weights (fp32 master) | 4 | 4.4 GB |
| Gradients (fp32) | 4 | 4.4 GB |
| Adam optimizer states (`m` and `v`, fp32) | 8 | 8.8 GB |
| **Fixed cost** | **16** | **~17.6 GB** |

This is the single most useful number in the whole repo. To size hardware for any model:

> **Pick GPUs where `16 × params` exceeds one card's VRAM but `16 × params ÷ N` does not.**
> Above the first threshold DDP is impossible; below the second, FSDP works.

### Memory Math with TinyLlama-1.1B on 2× 24GB GPUs

At batch size 8, sequence length 512:

| Component | DDP (per GPU) | FSDP (per GPU) |
|---|---|---|
| Fixed cost (weights + grads + optimizer) | ~17.6 GB | ~8.8 GB (÷2 GPUs) |
| Activations (bf16) | ~6.3 GB | ~6.3 GB |
| Logits + loss, CUDA context | ~2.1 GB | ~2.1 GB |
| **Total** | **~26 GB** ✗ | **~17 GB** ✓ |
| **Fits on a 24GB card?** | **No — OOM** | **Yes, ~7GB spare** |

> **Why this matters:** the fixed cost does not shrink when you lower the batch size — it is there before training starts. DDP spends 17.6GB of a 24GB card on it, leaving ~6GB for activations, which caps the batch at 2. FSDP spends 8.8GB, leaving ~15GB, which comfortably fits a batch of 8.

Activation figures are estimates and will vary with your transformers version and attention backend. If you land slightly over the limit, `gradient_checkpointing=True` in `src/train.py` trades ~30% speed for a large activation saving.

### How DDP Works

1. Each GPU loads a **full copy** of the model
2. Each GPU processes a different mini-batch (data parallelism)
3. After the backward pass, gradients are **all-reduced** (averaged) across GPUs
4. Each GPU updates its own copy of the model with the same averaged gradients

**Pros:** Simple, low communication overhead
**Cons:** Model must fit entirely on one GPU (with optimizer + gradients + activations)

### How FSDP Works

1. Model parameters are **sharded** — each GPU holds only 1/N of the parameters
2. Before each forward/backward pass, parameters are **all-gathered** (temporarily reassembled)
3. After computing gradients, they are **reduce-scattered** (averaged and re-sharded)
4. Each GPU only updates its own shard of the optimizer states

**Pros:** Dramatically lower per-GPU memory usage
**Cons:** Higher communication overhead (all-gather before each layer)

### When to Use What

| | DDP | FSDP |
|---|---|---|
| **Memory per GPU** | Full model + optimizer + gradients | 1/N of model + optimizer + gradients |
| **Communication** | All-reduce gradients (once per step) | All-gather params + reduce-scatter gradients (per layer) |
| **Best for** | Models that comfortably fit on a single GPU | Large models that need memory optimization |
| **Batch size** | Limited by leftover GPU memory | Much larger — sharding frees memory |
| **Overhead** | Lower | Higher (more communication) |
| **Complexity** | Simpler setup | More config options |

**Rule of thumb:** If your model + optimizer + gradients fit on one GPU with plenty of room for activations, DDP is simpler and faster. If you're memory-constrained, FSDP lets you trade communication overhead for larger batches (or train models that wouldn't fit at all with DDP).

## Accelerate Configuration

Accelerate configs live in `configs/` and control the distributed strategy.
You can either use the provided YAML files directly or generate your own via `accelerate config`.

Both configs use `mixed_precision: bf16`, matching `bf16=True` in `src/train.py`.
They must agree — HuggingFace Trainer errors out if accelerate and `TrainingArguments`
disagree about precision.

### DDP Config

`configs/ddp_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 2
mixed_precision: bf16
main_training_function: main
```

Key fields:
- `distributed_type: MULTI_GPU` — tells accelerate to use DDP
- `num_processes: 2` — number of GPUs (one process per GPU)
- `mixed_precision: bf16` — uses bfloat16 for the forward/backward compute

To generate interactively:
```bash
accelerate config --config_file configs/ddp_config.yaml
# Compute environment → This machine
# Distributed training → multi-GPU
# How many machines → 1
# How many GPUs → 2
# Mixed precision → bf16
```

### FSDP Config

`configs/fsdp_config.yaml`:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_machines: 1
num_processes: 2
mixed_precision: bf16
main_training_function: main
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
```

Key FSDP fields explained:

| Field | Value | What it does |
|---|---|---|
| `fsdp_sharding_strategy` | `FULL_SHARD` | Shards parameters, gradients, AND optimizer states (maximum memory savings) |
| `fsdp_auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` | Wraps each transformer layer as a separate FSDP unit |
| `fsdp_transformer_layer_cls_to_wrap` | `LlamaDecoderLayer` | The layer class to wrap — **must match your model architecture** |
| `fsdp_backward_prefetch_policy` | `BACKWARD_PRE` | Prefetches next layer's params during backward pass (faster) |
| `fsdp_state_dict_type` | `SHARDED_STATE_DICT` | Saves checkpoints in sharded format (avoids gathering full model to one GPU) |
| `fsdp_sync_module_states` | `true` | Ensures all GPUs start with identical weights |
| `fsdp_use_orig_params` | `true` | Required for compatibility with HuggingFace Trainer |
| `fsdp_cpu_ram_efficient_loading` | `true` | Loads model on CPU first, then distributes (avoids GPU OOM during init) |
| `fsdp_offload_params` | `false` | Set to `true` to offload params to CPU (slower but saves even more GPU memory) |

> **Important:** When switching models, you must update `fsdp_transformer_layer_cls_to_wrap` to match the new model's decoder layer class. TinyLlama is a Llama architecture, hence `LlamaDecoderLayer`. For example: `Qwen2DecoderLayer` for Qwen2.5 models, `MistralDecoderLayer` for Mistral, etc.

To generate interactively:
```bash
accelerate config --config_file configs/fsdp_config.yaml
# Compute environment → This machine
# Distributed training → FSDP
# How many machines → 1
# How many GPUs → 2
# Sharding strategy → FULL_SHARD
# Auto wrap policy → TRANSFORMER_BASED_WRAP
# Transformer layer class → LlamaDecoderLayer
# Backward prefetch → BACKWARD_PRE
# State dict type → SHARDED_STATE_DICT
# Mixed precision → bf16
```

## Run Training

### DDP (batch size = 2 per GPU)

```bash
accelerate launch --config_file configs/ddp_config.yaml train_ddp.py
```

Effective batch size: 2 × 4 (grad accum) × 2 (GPUs) = **16**

### FSDP (batch size = 8 per GPU)

```bash
accelerate launch --config_file configs/fsdp_config.yaml train_fsdp.py
```

Effective batch size: 8 × 4 (grad accum) × 2 (GPUs) = **64**

> **The key takeaway:** FSDP handles a 4× larger per-device batch size than DDP on the same hardware. DDP runs out of memory (OOM) at batch size 8 with TinyLlama-1.1B on 24GB GPUs, while FSDP handles it comfortably.

Each run prints its peak GPU memory per rank when training finishes, so you can
compare the two directly rather than taking the tables above on faith:

```
[rank 0] peak GPU memory: 20.1 GB / 24.0 GB     # DDP,  batch 2
[rank 0] peak GPU memory: 17.2 GB / 24.0 GB     # FSDP, batch 8
```

FSDP uses *less* memory while doing *4× more work per step* — that is the whole point.

### Experimenting with Batch Sizes

To see the OOM difference yourself, try editing the batch size in `train_ddp.py`:

```python
# In train_ddp.py — change this:
BATCH_SIZE = 2
# To this (will OOM on a 24GB card):
BATCH_SIZE = 8
```

You should see a CUDA out-of-memory error with DDP, proving that FSDP's memory savings are real and meaningful.

## Project Structure

```
├── configs/
│   ├── ddp_config.yaml       # Accelerate config for DDP
│   └── fsdp_config.yaml      # Accelerate config for FSDP
├── src/
│   ├── __init__.py
│   ├── data.py               # Dataset loading & Alpaca formatting
│   ├── train.py              # SFTTrainer training logic + hyperparameters
│   └── utils.py              # Peak GPU memory reporting
├── train_ddp.py              # DDP entry point (batch_size=2)
├── train_fsdp.py             # FSDP entry point (batch_size=8)
├── pyproject.toml            # Dependencies (managed by uv)
├── bash_command.md           # Setup runbook for a rented vast.ai box
└── README.md
```

## Configuration

Hyperparameters are defined in `src/train.py`. Batch size is set per-strategy in the entry point scripts.

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Model to finetune |
| `DEFAULT_BATCH_SIZE` | 2 | Per-device batch size (overridden by entry scripts) |
| `GRAD_ACCUM_STEPS` | 4 | Gradient accumulation steps |
| `MAX_SEQ_LENGTH` | 512 | Max token length |
| `NUM_SAMPLES` | 10,000 | Subset size from Alpaca dataset |
| `NUM_EPOCHS` | 1 | Number of training epochs |
| `LR` | 2e-5 | Learning rate |
| `save_model` | `False` | Whether to save the trained model and tokenizer to disk |

> **On `dtype=torch.float32`:** the model is deliberately loaded in fp32 while
> `bf16=True` is set. That combination is standard mixed precision — fp32 master
> weights and fp32 optimizer states, with bf16 used only for compute. Loading the
> model in bf16 instead would let AdamW keep bf16 optimizer states, halving the
> fixed cost to 8 bytes/param. Cheaper, but numerically worse *and* it would make
> DDP fit at batch size 8, quietly destroying the comparison this repo exists to show.

> **Disk space:** By default, model saving is disabled (`save_model=False`) and no checkpoints are written during training. To save the final model, pass `save_model=True` to the `train()` function in your entry script:
> ```python
> train(batch_size=BATCH_SIZE, save_model=True)
> ```

### Batch Sizes Per Strategy

| | Per-device | × Grad Accum | × GPUs | Effective |
|---|---|---|---|---|
| **DDP** | 2 | 4 | 2 | **16** |
| **FSDP** | 8 | 4 | 2 | **64** |

## Scaling Up

The configs here are sized for TinyLlama-1.1B on 2× 24GB. To run a larger model,
apply the 16 bytes/param rule and change three things:

1. `MODEL_NAME` in `src/train.py`
2. `fsdp_transformer_layer_cls_to_wrap` in `configs/fsdp_config.yaml` to match the new architecture
3. `num_processes` in both configs to match your GPU count

| Model | Fixed cost (16 B/param) | FSDP per GPU |
|---|---|---|
| TinyLlama-1.1B | ~17.6 GB | ~8.8 GB on 2 GPUs |
| Qwen2.5-1.5B | ~24 GB | ~12 GB on 2 GPUs |
| Llama-3.1-8B | ~128 GB | ~32 GB on 4 GPUs |
| Qwen2.5-7B | ~112 GB | ~28 GB on 4 GPUs |

---
