"""Dataset loading and formatting for SFTTrainer."""

from datasets import load_dataset
from transformers import AutoTokenizer


def format_alpaca(example):
    """Convert an alpaca example into a single text string."""
    if example["input"]:
        text = (
            f"Below is an instruction that describes a task, paired with further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataset(dataset_name, tokenizer, max_seq_length, num_samples):
    raw = load_dataset(dataset_name, split="train")
    raw = raw.select(range(min(num_samples, len(raw))))
    raw = raw.map(format_alpaca, remove_columns=raw.column_names)
    return raw
