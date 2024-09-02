import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base URL for the AJAX request
ajax_url = "https://www.addgene.org/search/catalog/plasmids/data/"

# Curation tags to scrape
curation_tags = [
    "CRISPR Mammalian Cut",
    "CRISPR Base Edit",
    "CRISPR Nick",
    "CRISPR Prime Edit",
    "CRISPR Activate",
    "CRISPR Interfere",
    "CRISPR Epigenetics",
    "CRISPR RNA Targeting",
    "CRISPR RNA Editing",
    "CRISPR Purify",
    "CRISPR Tag",
    "CRISPR Visualize",
    "CRISPR dCas9-FokI",
    "CRISPR Pooled Libraries",
    "CRISPR Empty gRNA Vectors",
    "CRISPR gRNAs"
]

# Parameters for the AJAX request
params = {
    'column': ['flame', 'item', 'insert', 'promoter', 'selectable_marker', 'pi', 'article']
}

# Headers to mimic a browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest"
}

def extract_data(item):
    flame_soup = BeautifulSoup(item['flame'], 'html.parser')
    plasmid_soup = BeautifulSoup(item['item'], 'html.parser')
    pi_soup = BeautifulSoup(item['pi'], 'html.parser')
    article_soup = BeautifulSoup(item['article'], 'html.parser')

    flame_class = flame_soup.find('span')['class'][2] if flame_soup.find('span') else 'N/A'
    plasmid_id = plasmid_soup.find('a')['href'].strip('/')
    plasmid_entry = {
        'Flame': flame_class,
        'Plasmid': plasmid_soup.get_text(strip=True),
        'Link': f"https://www.addgene.org/{plasmid_id}",
        'ID': plasmid_id,
        'Gene/Insert': item['insert'],
        'Promoter': item['promoter'],
        'Selectable Marker': item['selectable_marker'],
        'PI': pi_soup.get_text(strip=True),
        'PI Link': f"https://www.addgene.org{pi_soup.find('a')['href']}" if pi_soup.find('a') else 'N/A',
        'Publication': article_soup.get_text(strip=True),
        'Publication Link': f"https://www.addgene.org{article_soup.find('a')['href']}" if article_soup.find('a') else 'N/A'
    }
    return plasmid_entry

def fetch_plasmid_data(last_update=None):
    all_plasmid_data = []
    for tag in curation_tags:
        params['curation_tags'] = tag
        if last_update:
            params['last_update'] = last_update
        response = requests.get(ajax_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data['data']:
                plasmid_data = extract_data(item)
                plasmid_data['Curation Tag'] = tag  # Add the curation tag to the data
                all_plasmid_data.append(plasmid_data)
        else:
            logging.error(f"Failed to retrieve data for {tag}: {response.status_code}")
    return all_plasmid_data

def get_full_sequence(plasmid_url):
    sequence_url = plasmid_url + "/sequences"
    response = requests.get(sequence_url)
    logging.info(f"Response: {response.status_code}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        sequence_text = None
        sequence_source = None
        genbank_link = None

        depositor_section = soup.find('section', {'id': 'depositor-full'})
        if depositor_section:
            sequence_textarea = depositor_section.find('textarea', {'class': 'copy-from form-control'})
            if sequence_textarea:
                sequence_text = sequence_textarea.text.strip()
                sequence_source = 'Depositor'
            genbank_tag = depositor_section.find('a', {'class': 'genbank-file-download'})
            if genbank_tag:
                genbank_link = genbank_tag['href']

        addgene_section = soup.find('section', {'id': 'addgene-full'})
        if addgene_section:
            sequence_textarea = addgene_section.find('textarea', {'class': 'copy-from form-control'})
            if sequence_textarea:
                sequence_text = sequence_textarea.text.strip()
                sequence_source = 'Addgene'
            genbank_tag = addgene_section.find('a', {'class': 'genbank-file-download'})
            if genbank_tag:
                genbank_link = genbank_tag['href']

        if sequence_text:
            sequence_text_lines = sequence_text.split('\n')
            sequence_text = ''.join(line for line in sequence_text_lines if not line.startswith('>'))

        return sequence_text, sequence_source, genbank_link
    
    return None, None, None

def scrape_sequences(metadata):
    sequences = []
    sources = []
    genbank_links = []

    for index, row in metadata.iterrows():
        plasmid_url = row['Link']
        logging.info(f"Scraping sequence for: {plasmid_url}")
        sequence, source, genbank_link = get_full_sequence(plasmid_url)
        sequences.append(sequence)
        sources.append(source)
        genbank_links.append(genbank_link)
        time.sleep(0.1)  # Be polite and avoid overwhelming the server

    metadata['Full Sequence'] = sequences
    metadata['Sequence Source'] = sources
    metadata['GenBank Link'] = genbank_links

def download_genbank(args, output_dir):
    index, row = args
    genbank_url = row['GenBank Link']
    plasmid_id = row['ID']
    if pd.isna(genbank_url):
        return None
    
    response = requests.get(genbank_url)
    if response.status_code == 200:
        genbank_dir = os.path.join(output_dir, 'genbank_files')
        os.makedirs(genbank_dir, exist_ok=True)
        filename = f"plasmid_{plasmid_id}.gb"
        filepath = os.path.join(genbank_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return filepath
    return None

def main(output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    last_update_file = os.path.join(output_dir, 'last_update.txt')
    if os.path.exists(last_update_file):
        with open(last_update_file, 'r') as f:
            last_update = f.read().strip()
    else:
        last_update = None

    logging.info("Fetching plasmid data...")
    all_plasmid_data = fetch_plasmid_data(last_update)

    if not all_plasmid_data:
        logging.info("No new data found.")
        return

    metadata = pd.DataFrame(all_plasmid_data)
    metadata_file = os.path.join(output_dir, 'addgene_metadata.csv')

    logging.info("Scraping sequences...")
    print(metadata.head())
    print(metadata.shape)
    scrape_sequences(metadata)
    metadata.to_csv(metadata_file, index=False)
    logging.info(f"Sequences scraped and data saved to {metadata_file}")

    logging.info("Downloading GenBank files...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        genbank_files = list(tqdm(executor.map(download_genbank, enumerate(metadata.to_dict('records')), [output_dir]*len(metadata)), total=len(metadata)))

    metadata['GenBank File'] = genbank_files
    metadata.to_csv(metadata_file, index=False)
    logging.info(f"Updated metadata saved to {metadata_file}")

    with open(last_update_file, 'w') as f:
        f.write(pd.Timestamp.now().isoformat())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape plasmid data and sequences.')
    parser.add_argument('output_dir', type=str, help='Directory to save output files')
    args = parser.parse_args()

    main(output_dir=args.output_dir)