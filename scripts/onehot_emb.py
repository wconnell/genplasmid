from datasets import load_dataset
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
from genplasmid.datasets import genbank_to_glm2, read_genbank
import warnings
from datasets import load_dataset

from Bio import BiopythonParserWarning

import torch
import transformers
from transformers import PreTrainedTokenizerFast, GenerationConfig, AutoModel
from huggingface_hub import hf_hub_download
import numpy as np
import os

from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation
import warnings
from Bio import BiopythonParserWarning

from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
import pandas as pd
from itertools import product
import pyarrow.parquet as pq
import pyarrow as pa

import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix, csr_matrix, dok_matrix
from itertools import product

import pickle

from pathlib import Path
DATA_PATH = Path("./data")
NUCLEOTIDE_PATH = DATA_PATH / "nucleotide_sequence.parquet"
ONEHOT_PATH = DATA_PATH / "onehot_embedding_dataframe.pkl"
KMER_PATH = DATA_PATH / "onehot_kmer_{}"


def genbank_to_plasmidgpt(record: str) -> str:
    if record == '':
        return None
    
    genbank_record = SeqIO.read(StringIO(record), "genbank")
    sequence = str(genbank_record.seq).upper()
    return sequence

def one_hot_encode_sequence(sequence, vocab_dict):
    # Initialize empty matrix for one-hot encoding
    vocab_size = len(vocab_dict)
    one_hot_matrix = np.zeros((len(sequence), vocab_size), dtype=np.int8)

    # Encode each nucleotide
    for idx, nucleotide in enumerate(sequence):
        if nucleotide in vocab_dict:
            one_hot_matrix[idx, vocab_dict[nucleotide]] = 1
    
    return one_hot_matrix

def generate_kmer_vocab(k):
    nucleotides = ['A', 'C', 'G', 'T']

    kmer_vocab = []
    for i in range(2, k+1): 
        kmer_list = [''.join(kmer) for kmer in product(nucleotides, repeat=i)]
        kmer_vocab.extend(kmer_list)

    kmer_vocab += nucleotides    
    return kmer_vocab

def one_hot_encode_kmers(sequence, k, kmer_dict):
    kmer_size = len(kmer_dict)
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return one_hot_encode_sequence(kmers, kmer_dict)

def load_nucleotide_data():
    if os.path.exists(NUCLEOTIDE_PATH):
        df = pd.read_parquet(NUCLEOTIDE_PATH)
        return df

    data = load_dataset("wconnell/openplasmid")
    data = data.filter(lambda x: x['GenBank Raw'] != '')
    data = data.map(lambda x: {'nucleotide_sequence': genbank_to_plasmidgpt(x['GenBank Raw'])})
    df = data['train'].to_pandas()
    df = df[['ID', 'nucleotide_sequence']]
    
    df.to_parquet(NUCLEOTIDE_PATH)
    return df

def compute_onehot(df): 
    if os.path.exists(ONEHOT_PATH):
        return 

    print('One-hot encoding nucleotide sequences...')
    vocab_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    vocab_path = ONEHOT_PATH.parent / 'vocab_dict.pkl'
    import pickle
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_dict, f)

    onehot_sequence = df['nucleotide_sequence'].apply(lambda x: one_hot_encode_sequence(x, vocab_dict))
    # get onehot sequence as np array
    onehot_sequence = onehot_sequence.to_numpy()
    emb_df = pd.DataFrame(onehot_sequence, index=df.ID, columns=['embedding']) 

    import pickle 
    with open(ONEHOT_PATH, 'wb') as f:
        pickle.dump(emb_df, f)

def compute_kmers(df, k):
    if os.path.exists(KMER_PATH):
        return 

    KMER_PATH.mkdir(exist_ok=True)
    
    print('One-hot encoding k-mer sequences...')
    kmer_vocab = generate_kmer_vocab(3)
    kmer_dict = {kmer: idx for idx, kmer in enumerate(kmer_vocab)}
    
    import pickle 
    with open(KMER_PATH / 'vocab_dict.pkl', 'wb') as f:
        pickle.dump(kmer_dict, f)

    # chunk df into batches
    batch_size = 4096
    num_batches = int(np.ceil(len(df) / batch_size))

    from tqdm import tqdm
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start:end]
        kmer_onehot = batch['nucleotide_sequence'].apply(lambda x: one_hot_encode_kmers(x, k, kmer_dict))
        # get onehot sequence as np array
        combined_kmer_matrix = kmer_onehot.to_numpy()
        emb_df = pd.DataFrame(combined_kmer_matrix, index=batch.ID, columns=['embedding'])
        filename = KMER_PATH / f'{start}_{end}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(emb_df, f)


def main(): 
    import os 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    DATA_PATH.mkdir(exist_ok=True)
    global KMER_PATH
    KMER_PATH = Path(str(KMER_PATH).format(args.k))

    df = load_nucleotide_data()
    
    try: 
        compute_onehot(df)
    except Exception as e:
        print(e)

    try: 
        compute_kmers(df, k=args.k)
        
    except Exception as e:
        print(e)



if __name__ == "__main__":
    main()