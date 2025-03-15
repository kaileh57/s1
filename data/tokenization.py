from typing import Dict
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    return text

def process_claude_example(
    example: Dict,
    tokenizer,
):
    # Use Claude thinking trajectory instead of Gemini's
    thinking_trajectory = example["claude_thinking_trajectory"]
    question = example["question"]
    answer = example["claude_attempt"] 
    
    # Use Claude-style <think> tags
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": question},
        {
            "role": "assistant", 
            "content": f"<think>{thinking_trajectory}</think>{answer.strip()}"
        }
    ], tokenize=False)
    
    return dict(text=text)

def mathcot_sft(upload_data_path: str, num_proc: int,
                download_data_path):

    dataset = load_dataset(download_data_path, download_mode='force_redownload')
    if 'train' in dataset:
        dataset = dataset['train']
    
    # Use Gemma tokenizer instead of Qwen
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
    process_example_map = partial(process_claude_example, tokenizer=tokenizer)
    
    # Create train/test split for evaluation
    split_dataset = dataset.train_test_split(test_size=0.1)
    
    train_tokenized = split_dataset["train"].map(
        process_example_map,
        num_proc=num_proc,
        desc="Tokenizing training data",
    )
    
    test_tokenized = split_dataset["test"].map(
        process_example_map,
        num_proc=num_proc,
        desc="Tokenizing test data",
    )
    
    # Combine into a DatasetDict
    from datasets import DatasetDict
    final_dataset = DatasetDict({
        "train": train_tokenized,
        "test": test_tokenized
    })
    
    final_dataset.push_to_hub(upload_data_path)

if __name__ == "__main__":
    mathcot_sft(download_data_path="simplescaling/s1K-claude-3-7-sonnet",
                upload_data_path="Sculptor-AI/s1K-claude-gemma-tokenized", 
                num_proc=8)