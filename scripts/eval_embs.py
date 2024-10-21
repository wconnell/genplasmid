import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from genplasmid.datasets import genbank_to_glm2, read_genbank
import warnings
from datasets import load_dataset
from Bio import BiopythonParserWarning
import re
from collections import Counter
import scanpy as sc
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=BiopythonParserWarning, message="Attempting to parse malformed locus line:")

def load_and_preprocess_data():
    data = load_dataset("wconnell/openplasmid")
    data = data.filter(lambda x: x['GenBank Raw'] != '')
    data = data.map(lambda x: {'glm2_sequence': genbank_to_glm2(x['GenBank Raw'])})

    all_feat = data['train'].to_pandas()
    all_feat['GenBank'] = all_feat['GenBank Raw'].map(read_genbank)
    return all_feat

def clean_gene_name(gene):
    gene = re.sub(r'^(human|mouse|rat|h|m|r)\s*', '', gene, flags=re.IGNORECASE)
    gene = re.sub(r'\s*(gene|protein)$', '', gene, flags=re.IGNORECASE)
    gene = re.sub(r'\s*\([^)]*\)', '', gene)
    
    gene_map = {
        'neo': 'neomycin resistance',
        'amp': 'ampicillin resistance',
        'gfp': 'GFP',
        'egfp': 'GFP',
        'rfp': 'RFP',
        'dsred': 'RFP',
        'kan': 'kanamycin resistance',
    }
    
    for key, value in gene_map.items():
        if re.search(rf'\b{key}\b', gene, re.IGNORECASE):
            return value
    
    return gene.strip().lower()

def extract_cds_genes(record):
    return [clean_gene_name(feature.qualifiers.get('gene', feature.qualifiers.get('product', ['']))[0])
            for feature in record.features if feature.type == 'CDS']

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if sequence else 0

def extract_sequence_features(all_feat):
    all_feat['CDS genes'] = all_feat['GenBank'].map(extract_cds_genes)
    all_feat['Sequence length'] = all_feat['GenBank'].map(lambda x: len(x.seq))
    all_feat['log(seq_len)'] = np.log10(all_feat['Sequence length'])
    all_feat['GC content'] = all_feat['GenBank'].map(lambda x: calculate_gc_content(str(x.seq)))
    return all_feat

def define_keywords(all_feat):
    feature_counts = Counter(gene for genes in all_feat['CDS genes'] for gene in genes)
    exclude = ['bla', 'op']
    return [key for key, _ in feature_counts.most_common(15) if key not in exclude][::-1]

def map_genes(genes, keywords):
    return next((keyword for keyword in keywords if any(re.search(keyword, gene, re.IGNORECASE) for gene in genes)), pd.NA)

def process_entrez_genes(all_feat):
    all_feat['Entrez Genes'] = all_feat.apply(lambda row: [insert['Entrez Gene'].upper() 
                                                           for i in range(1, 4) 
                                                           for insert in [row[f'Gene/Insert {i}']] 
                                                           if isinstance(insert, dict) and insert.get('Entrez Gene')], axis=1)

    common_entrez = all_feat['Entrez Genes'].explode().value_counts().head(20).index.tolist()
    common_entrez_priority = {gene: i for i, gene in enumerate(reversed(common_entrez))}

    all_feat['common-entrez-gene'] = all_feat['Entrez Genes'].apply(lambda genes: max(
        (gene for gene in genes if gene in common_entrez_priority),
        key=lambda g: common_entrez_priority[g],
        default=pd.NA
    ))
    return all_feat

def create_anndata(embeddings, all_feat):
    adata = sc.AnnData(embeddings, obs=all_feat)
    sc.tl.pca(adata)
    return adata

import textwrap

def wrap_labels(label, width=20):
    return '\n'.join(textwrap.wrap(label, width=width))

def perform_clustering(adata, feature_column, results_dir, metrics_df):
    # Wrap the labels in the adata object, excluding NaN and None values
    adata.obs[feature_column] = adata.obs[feature_column].map(lambda x: wrap_labels(x, width=20) if pd.notna(x) else x)
    # Plot the full adata
    plt.figure(figsize=(16, 10))  # Keep existing figure size
    sc.pl.pca(
        adata,
        color=[feature_column],
        ncols=1,
        legend_fontsize='xx-small',
        title=[f"Full {feature_column}"],
        show=False,
        na_color="lightgrey"
    )
    plt.tight_layout()  # Keep existing layout adjustment
    plt.savefig(os.path.join(results_dir, f"clustering_full_{feature_column}.png"), bbox_inches='tight', dpi=300)
    plt.close()

    filtered_adata = adata[adata.obs[feature_column].notna()]
    n_clusters = len(filtered_adata.obs[feature_column].unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(filtered_adata.obsm['X_pca'])
    feature_labels = filtered_adata.obs[feature_column].astype('category').cat.codes

    nmi_score = normalized_mutual_info_score(feature_labels, cluster_labels)
    ari_score = adjusted_rand_score(feature_labels, cluster_labels)

    print(f"Clustering results for {feature_column}:")
    print(f"Normalized Mutual Information: {nmi_score:.4f}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")

    new_row = pd.DataFrame({
        'Feature': [feature_column],
        'NMI': [nmi_score],
        'ARI': [ari_score]
    })
    
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    plt.figure(figsize=(16, 10))  # Keep existing figure size
    sc.pl.pca(
        filtered_adata,
        color=[feature_column],
        ncols=1,
        legend_fontsize='xx-small',
        title=[f"Filtered {feature_column}"],
        show=False
    )
    plt.tight_layout()  # Keep existing layout adjustment
    plt.savefig(os.path.join(results_dir, f"clustering_filtered_{feature_column}.png"), bbox_inches='tight', dpi=300)
    plt.close()

    return metrics_df

def main(embeddings_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../results/eval/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    all_feat = load_and_preprocess_data()
    all_feat = extract_sequence_features(all_feat)
    
    keywords = define_keywords(all_feat)
    all_feat['CDS-curated-features'] = all_feat['CDS genes'].apply(lambda x: map_genes(x, keywords))
    
    all_feat = process_entrez_genes(all_feat)
    
    embeddings = pd.read_parquet(embeddings_file)
    embeddings.index = embeddings.index.astype(int)
    embeddings = embeddings.dropna()

    # align metadata with embeddings
    all_feat = all_feat.set_index('ID')
    all_feat.index = all_feat.index.astype(int)
    all_feat = all_feat.loc[embeddings.index]
    
    if len(embeddings) != len(all_feat):
        raise ValueError("Number of embeddings does not match number of samples in all_feat")
    
    adata = create_anndata(embeddings.to_numpy(), all_feat)
    
    metrics_df = pd.DataFrame(columns=['Feature', 'NMI', 'ARI'])

    for feature in ['CDS-curated-features', 'common-entrez-gene']:
        metrics_df = perform_clustering(adata, feature, results_dir, metrics_df)
    
    metrics_df.to_csv(os.path.join(results_dir, "clustering_metrics.csv"), index=False)

    print(f"Results saved in directory: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embeddings for plasmid sequences")
    parser.add_argument("embeddings_file", help="Path to the parquet file containing embeddings. Index should be 'ID'.")
    args = parser.parse_args()
    
    main(args.embeddings_file)