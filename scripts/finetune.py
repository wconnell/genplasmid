import os
import argparse
import logging
import json
from datetime import datetime
import uuid
import shutil
import torch
from datasets import load_dataset
from accelerate import find_executable_batch_size
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

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

def tokenize_and_chunk(examples, tokenizer):
    tokenized = tokenizer(examples['glm2_sequence'], truncation=False, padding=False)
    result = {'input_ids': [], 'attention_mask': []}
    
    for ids, mask in zip(tokenized['input_ids'], tokenized['attention_mask']):
        if len(ids) == 0:  # Check for empty sequences
            logging.warning("Empty sequence encountered, skipping.")
            continue
        
        for i in range(0, len(ids), 4096):
            chunk_ids = ids[i:i+4096]
            chunk_mask = mask[i:i+4096]
            
            # Pad if necessary
            if len(chunk_ids) < 4096:
                padding_length = 4096 - len(chunk_ids)
                chunk_ids = chunk_ids + [tokenizer.pad_token_id] * padding_length
                chunk_mask = chunk_mask + [0] * padding_length
            
            # Check for valid chunk sizes
            if len(chunk_ids) != 4096 or len(chunk_mask) != 4096:
                logging.warning(f"Invalid chunk size: {len(chunk_ids)} for ids or {len(chunk_mask)} for mask, skipping.")
                continue
            
            result['input_ids'].append(chunk_ids)
            result['attention_mask'].append(chunk_mask)
    
    return result

def find_batch_size(starting_batch_size, model, tokenizer, train_dataset):
    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def train_batch(batch_size):
        batch_size = max(1, batch_size)  # Ensure batch_size is at least 1
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

def main(num_train_epochs, starting_batch_size, gradient_accumulation_steps):
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("..", "results", "finetune", f"{timestamp}-{run_id}")
    os.makedirs(results_dir, exist_ok=True)

    setup_logging(results_dir)
    logging.info(f"Starting fine-tuning script with run ID: {run_id}")
    logging.info(f"Arguments: num_train_epochs={num_train_epochs}, starting_batch_size={starting_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")

    logging.info("Loading and preparing dataset")
    data = load_dataset("wconnell/openplasmid")['train']
    # data = data.select(range(100))
    data = data.filter(lambda x: x['GenBank Raw'] != '')
    data = data.map(lambda x: {'glm2_sequence': genbank_to_glm2(x['GenBank Raw'])})
    data = data.filter(lambda x: x['glm2_sequence'] != '')
    data = data.train_test_split(test_size=0.2, seed=42)  # Added seed for reproducibility

    logging.info("Loading pre-trained model and tokenizer")
    model = AutoModelForMaskedLM.from_pretrained('tattabio/gLM2_150M', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_150M', trust_remote_code=True)

    logging.info("Tokenizing and chunking dataset")
    data = data.map(
        lambda examples: tokenize_and_chunk(examples, tokenizer),
        batched=True,
        remove_columns=data['train'].column_names
    )   
    logging.info(f"Train dataset size: {len(data['train'])}")
    logging.info(f"Test dataset size: {len(data['test'])}")

    data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    logging.info("Finding executable batch size")
    executable_batch_size = find_batch_size(args.starting_batch_size, model, tokenizer, data['train'])
    
    if os.path.exists("./temp_output"):
        shutil.rmtree("./temp_output")
    logging.info(f"Found executable batch size: {executable_batch_size}")

    training_args = TrainingArguments(
        output_dir=os.path.join(results_dir, "checkpoints"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=executable_batch_size,
        per_device_eval_batch_size=executable_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=5,
        logging_dir=os.path.join(results_dir, "logs"),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        evaluation_strategy="epoch",  # Added this line
        load_best_model_at_end=True,  # Added this line
        metric_for_best_model="eval_loss",  # Added this line
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
    )

    logging.info("Starting fine-tuning")
    trainer.train()

    final_model_path = os.path.join(results_dir, "final-model")
    trainer.save_model(final_model_path)  # Changed this line
    logging.info(f"Fine-tuned model saved to {final_model_path}")

    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "args": {
            "num_train_epochs": num_train_epochs,
            "starting_batch_size": starting_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps
        },
        "model_name": "gLM2_150M_finetuned",
        "final_model_path": final_model_path,
        "executable_batch_size": executable_batch_size,
    }
    save_config(config, results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune gLM2 model")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--starting_batch_size", type=int, default=8, help="Starting batch size for processing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5, help="Number of gradient accumulation steps")
    args = parser.parse_args()

    main(args.num_train_epochs, args.starting_batch_size, args.gradient_accumulation_steps)