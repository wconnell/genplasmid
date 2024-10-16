import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
from genplasmid.dataset import genbank_to_glm2, read_genbank
import warnings
from datasets import load_dataset
from Bio import BiopythonParserWarning
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download, snapshot_download
import os

# Suppress the specific warning
warnings.filterwarnings("ignore", category=BiopythonParserWarning, message="Attempting to parse malformed locus line:")

PATH = snapshot_download("lingxusb/PlasmidGPT", cache_dir="./plasmidgpt/prediction_models")


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return torch.device(device)


DEVICE = get_device()


def load_model() -> torch.nn.Module:
    path = os.path.join(PATH, "pretrained_model.pt")
    model = torch.load(path, map_location=DEVICE)
    return model


def get_tokenizer():
    path = os.path.join(PATH, "addgene_trained_dna_tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
    special_tokens_dict = {'additional_special_tokens': ['[PROMPT]', '[PROMPT2]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.vocab['[PAD]']
    return tokenizer


def embed(sequence: str, model, tokenizer, average_hidden_states=True) -> torch.Tensor:
    input_ids = tokenizer.encode(sequence, return_tensors='pt').to(DEVICE)

    # provided in the plasmidgpt example notebook
    # note that [SEP] = 2 and [PAD] = 3
    special_tokens = torch.tensor([3] * 10 + [2], dtype=torch.long, device=DEVICE)
    input_ids = torch.cat((special_tokens.unsqueeze(0), input_ids), dim=1)

    batch = {'input_ids': input_ids, 'attention_mask': None}

    return embed_batch(batch, model, tokenizer, average_hidden_states)


def embed_batch(batch, model, average_hidden_states=True) -> torch.Tensor:
    model.config.output_hidden_states = True
    with torch.no_grad():
        batch_input_ids = batch['input_ids'].to(DEVICE)
        batch_attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        hidden_states = outputs.hidden_states[-1].cpu()

        if average_hidden_states:
            hidden_states = torch.mean(hidden_states, axis=1)

        return hidden_states


def embed_dataloader(dataloader, model, average_hidden_states=True):
    embeddings = []
    for batch in tqdm(dataloader):
        batch_embeddings = embed_batch(batch, model, average_hidden_states=average_hidden_states)
        embeddings.extend(batch_embeddings.numpy())

    return np.array(embeddings)


def genbank_to_plasmidgpt(record: str) -> str:
    if record == '':
        return None

    genbank_record = SeqIO.read(StringIO(record), "genbank")
    sequence = str(genbank_record.seq).upper()
    return sequence


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['plasmidgpt_sequence'],
        truncation=True,
        max_length=2048,
        padding='max_length',
    )


def generate_embeddings(dataloader, model, average_hidden_states=True):
    embeddings = []
    for batch in tqdm(dataloader):
        batch_embeddings = embed_batch(batch, model, average_hidden_states=average_hidden_states)
        embeddings.extend(batch_embeddings.numpy())
    return np.array(embeddings)


def main():
    model = load_model()
    tokenizer = get_tokenizer()

    print("Loading and preparing dataset...")

    data = load_dataset("wconnell/openplasmid")
    data = data.filter(lambda x: x['GenBank Raw'] != '')
    data = data.map(lambda x: {'plasmidgpt_sequence': genbank_to_plasmidgpt(x['GenBank Raw'])})

    data = data.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(data['train'], batch_size=32, shuffle=False)

    print("Generating embeddings...")
    embeddings = generate_embeddings(dataloader, model, tokenizer)

    print(f"Train embeddings shape: {embeddings.shape}")
    if not os.path.exists('/data/scratch/bty174/genplasmid/data'):
        os.makedirs('/data/scratch/bty174/genplasmid/data')

    ids = data['train']['ID']

    embeddings_df = pd.DataFrame(embeddings, index=ids)

    embeddings_df.to_parquet('/data/scratch/bty174/genplasmid/data/plasmidgpt_embeddings.parquet')


if __name__ == "__main__":
    main()
