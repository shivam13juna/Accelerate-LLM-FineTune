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
    """Load the tokenizer and make it usable for batched training.

    Args:
        model_name: HF Hub repo id, or a local path to a saved tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama tokenizers ship without a pad token, but batching needs one to fill
    # short sequences. Reusing EOS is the standard workaround — the attention
    # mask stops the model from attending to the padding either way.
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataset(dataset_name, tokenizer, max_seq_length, num_samples):
    """Load, subset and format the training data.

    Returns rows with a single "text" column. Tokenization is deliberately left
    to SFTTrainer, which is why two arguments here go unused.

    Args:
        dataset_name: HF Hub dataset id.
        tokenizer: Unused — SFTTrainer tokenizes the "text" column itself.
        max_seq_length: Unused — truncation comes from max_length in SFTConfig.
        num_samples: Cap on rows kept, so a run finishes in minutes rather than
            hours. Lower it to ~200 for a smoke test.
    """
    raw = load_dataset(dataset_name, split="train")  # Alpaca has no test split
    # min() guards against asking for more rows than the dataset holds.
    raw = raw.select(range(min(num_samples, len(raw))))
    # remove_columns drops the original instruction/input/output fields, leaving
    # only "text" — SFTTrainer would otherwise try to interpret the extra columns.
    raw = raw.map(format_alpaca, remove_columns=raw.column_names)
    return raw
