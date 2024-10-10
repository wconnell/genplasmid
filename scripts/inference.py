import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator, find_executable_batch_size
from tqdm import tqdm
import pandas as pd
from genplasmid.datasets import genbank_to_glm2
import argparse
import os
from datetime import datetime
from safetensors import safe_open
import json
import logging
import sys
import uuid

def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'inference.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def convert_id(example):
    id_value = example['ID']
    if isinstance(id_value, list):
        id_value = id_value[0] if id_value else None
    try:
        return {'ID': float(id_value) if id_value is not None else None}
    except (ValueError, TypeError):
        return {'ID': None}

def load_model(model_path=None):
    model = AutoModel.from_pretrained('tattabio/gLM2_150M', torch_dtype=torch.bfloat16, trust_remote_code=True) # TODO: can also use 650M
    
    if model_path:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        
        # Remove the 'glm2.' prefix from the keys if present
        state_dict = {k.replace('glm2.', ''): v for k, v in state_dict.items()}
        
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print("Unexpected keys:", incompatible_keys.unexpected_keys)
        print("Missing keys:", incompatible_keys.missing_keys)
    
    return model

def generate_embeddings_accel(accelerator, model, data, starting_batch_size):
    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inner_embedding_loop(batch_size):
        nonlocal accelerator, model, data
        accelerator.free_memory()
        
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        model, dataloader = accelerator.prepare(model, dataloader)
        
        embeddings = []
        ids = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings", disable=not accelerator.is_local_main_process):
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).float()
                embeddings.extend(accelerator.gather(batch_embeddings).cpu().numpy())
                ids.extend(accelerator.gather(batch['ID']).cpu().numpy())
        
        return embeddings, ids, batch_size  # Return the batch_size as well

    return inner_embedding_loop()

def save_config(config, results_dir):
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configuration saved to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with adaptive batch size")
    parser.add_argument("--starting_batch_size", type=int, default=8, help="Starting batch size for processing (default: 8)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the finetuned model safetensors file")
    args = parser.parse_args()

    accelerator = Accelerator()

    # Generate a unique run ID
    run_id = str(uuid.uuid4())

    # Create timestamped results directory only on the main process
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("..", "results", "inference", f"{timestamp}-{run_id}")
        os.makedirs(results_dir, exist_ok=True)

        # Setup logging
        setup_logging(results_dir)

        # Log the start of the script and the arguments
        logging.info(f"Starting inference script with run ID: {run_id}")
        logging.info(f"Command-line arguments: {vars(args)}")

    # Synchronize all processes
    accelerator.wait_for_everyone()

    # Load model and tokenizer
    model = load_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_150M', trust_remote_code=True)

    # Load and prepare data
    data = load_dataset("wconnell/openplasmid")['train']
    data = data.filter(lambda x: x['GenBank Raw'] != '')
    data = data.map(lambda x: {'glm2_sequence': genbank_to_glm2(x['GenBank Raw'])})
    data = data.map(lambda examples: tokenizer(examples['glm2_sequence'], truncation=True, padding='max_length', max_length=4096), batched=True)
    data = data.map(convert_id)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'ID'])

    # Generate embeddings
    embeddings, ids, used_batch_size = generate_embeddings_accel(accelerator, model, data, args.starting_batch_size)

    # Save results only on the main process
    if accelerator.is_main_process:
        total_samples = len(data)
        embeddings = embeddings[:total_samples]
        ids = ids[:total_samples]

        df = pd.DataFrame(embeddings, index=ids)
        df.index.name = 'ID'
        df.index = df.index.astype(int)

        model_name = "glm2v2_ft" if args.model_path else "glm2v2"
        output_file = os.path.join(results_dir, f"{model_name}_embeddings.parquet")
        df.to_parquet(output_file)

        logging.info(f"Generated embeddings for {len(df)} samples")
        logging.info(f"Results saved in: {output_file}")
        logging.info(f"Used batch size: {used_batch_size}")

        # Save configuration
        config = {
            "run_id": run_id,
            "timestamp": timestamp,
            "args": vars(args),
            "model_name": model_name,
            "num_samples": len(df),
            "embedding_dim": df.shape[1],
            "output_file": output_file,
            "used_batch_size": used_batch_size  # Add the used batch size to the config
        }
        save_config(config, results_dir)

if __name__ == "__main__":
    main()
    # run with: accelerate launch inference.py
