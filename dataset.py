import argparse
import random

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer

RANDOM_SEED = 137

N_SAMPLES = 100
N_EPOCHS = 5


def get_dataset_sample(size: int = N_SAMPLES):
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:10%]")

    random.seed(RANDOM_SEED)
    random_indices = random.sample(range(len(dataset["text"])), size)
    random_subset = [dataset["text"][i] for i in random_indices]
    return random_subset


def get_tokenizer() -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(random_subset, tokenizer: PreTrainedTokenizer):
    tokenized_data = tokenizer(random_subset, return_tensors="pt", truncation=True, padding=True)
    input_ids = tokenized_data["input_ids"]
    return input_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="The output file path where the tokenized dataset sample will be saved.")
    parser.add_argument("--size", type=int, default=N_SAMPLES, help="The size of the dataset sample.")
    args = parser.parse_args()

    dataset = get_dataset_sample(args.size)
    tokenizer = get_tokenizer()
    input_ids = tokenize_dataset(dataset, tokenizer)

    vocab_size = len(tokenizer)
    torch.save({"input_ids": input_ids, "vocab_size": vocab_size}, args.output)
