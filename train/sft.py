import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl

@dataclass
class TrainingConfig:
    # Changed model to Gemma 3 12B
    model_name: str = field(default="google/gemma-3-12b-it")
    block_size: int = field(default=16384)
    # Update for your organization
    wandb_project: Optional[str] = field(default="gemma-claude-s1")
    wandb_entity: Optional[str] = field(default="sculptor-ai") 
    # Update for your tokenized dataset
    train_file_path: Optional[str] = field(default='Sculptor-AI/s1K-claude-gemma-tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model - adjusted for Gemma 3
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto", 
        torch_dtype="auto",
        attn_implementation="flash_attention_2", 
        use_cache=False
    )

    dataset = load_dataset(config.train_file_path)

    # setting up trainer with Gemma tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    # Update for Gemma chat format 
    instruction_template = "<start_of_turn>user"
    response_template = "<start_of_turn>model"
    
    # Use a token that is in Gemma vocabulary
    tokenizer.pad_token = "</s>"

    # Only compute loss over assistant responses
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()