import os
import argparse
import logging
import json
from datetime import datetime
import uuid
import shutil

import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from accelerate import find_executable_batch_size

from genplasmid.datasets import genbank_to_glm2

def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'finetune.log')),
            logging.StreamHandler()
        ]
    )

def save_config(config, results_dir):
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configuration saved to {config_path}")

def find_batch_size(starting_batch_size, model, tokenizer, train_dataset):
    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def train_batch(batch_size):
        try:
            args = TrainingArguments(
                output_dir="./temp_output",
                per_device_train_batch_size=batch_size,
                max_steps=1,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
            )
            trainer.train()
            return True
        except Exception as e:
            logging.warning(f"Batch size {batch_size} failed with error: {str(e)}")
            return False

    return train_batch()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune gLM2 model")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--starting_batch_size", type=int, default=1, help="Starting batch size for processing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5, help="Number of gradient accumulation steps")
    args = parser.parse_args()

    # Generate a unique run ID
    run_id = str(uuid.uuid4())

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("/scratch/wconnell/genplasmid", "results", "finetune", f"{timestamp}-{run_id}")
    os.makedirs(results_dir, exist_ok=True)

    # Setup logging
    setup_logging(results_dir)

    # Log the start of the script and the arguments
    logging.info(f"Starting fine-tuning script with run ID: {run_id}")
    logging.info(f"Command-line arguments: {vars(args)}")

    # Load and prepare data
    logging.info("Loading and preparing dataset")
    data = load_dataset("wconnell/openplasmid")['train']
    # subset for testing
    data = data.select(range(1000))
    data = data.filter(lambda x: x['GenBank Raw'] != '')
    data = data.train_test_split(test_size=0.2)
    data = data.map(lambda x: {'glm2_sequence': genbank_to_glm2(x['GenBank Raw'])})
    data = data.filter(lambda x: x['glm2_sequence'] != '')

    # Load the pre-trained model and tokenizer
    logging.info("Loading pre-trained model and tokenizer")
    model = AutoModelForMaskedLM.from_pretrained('tattabio/gLM2_150M', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_150M', trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(examples['glm2_sequence'], truncation=True, padding='max_length', max_length=4096)

    # Apply tokenization to the datasets
    logging.info("Tokenizing dataset")
    data = data.map(tokenize_function, batched=True)

    # Set the format of the datasets to return PyTorch tensors
    data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Find executable batch size
    logging.info("Finding executable batch size")
    executable_batch_size = find_batch_size(args.starting_batch_size, model, tokenizer, data['train'])
    if os.path.exists("./temp_output"):
        shutil.rmtree("./temp_output") 
    logging.info(f"Found executable batch size: {executable_batch_size}")

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(results_dir, "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=executable_batch_size,
        per_device_eval_batch_size=executable_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=50_000,
        save_total_limit=5,
        logging_dir=os.path.join(results_dir, "logs"),
        logging_steps=1,
        eval_strategy="epoch",
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
    )

    # Fine-tune the model
    logging.info("Starting fine-tuning")
    trainer.train()

    # Save the fine-tuned model
    final_model_path = os.path.join(results_dir, "final-model")
    model.save_pretrained(final_model_path)
    logging.info(f"Fine-tuned model saved to {final_model_path}")

    # Save configuration
    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "args": vars(args),
        "model_name": "gLM2_150M_finetuned",
        "final_model_path": final_model_path,
        "executable_batch_size": executable_batch_size,
    }
    save_config(config, results_dir)

if __name__ == "__main__":
    main()