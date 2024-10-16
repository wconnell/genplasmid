from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation
import warnings
from Bio import BiopythonParserWarning

# Suppress the specific warning
warnings.filterwarnings("ignore", category=BiopythonParserWarning, message="Attempting to parse malformed locus line:")

def read_genbank(record: str):
    return SeqIO.read(StringIO(record), "genbank")

def genbank_to_glm2(record: str) -> str:
    if record == '':
        return None
    genbank_record = SeqIO.read(StringIO(record), "genbank")

    sequence = str(genbank_record.seq).lower()
    strand_map = {1: "<+>", -1: "<->"}
    formatted_sequence = []
    circular_features = []

    # Sort features by start position, then by end position in reverse
    features = sorted(genbank_record.features, key=lambda x: (x.location.start, -x.location.end))

    # Filter out 'source' feature and remove duplicates
    unique_features = []
    for feature in features:
        if feature.type != 'source' and feature not in unique_features:
            unique_features.append(feature)

    for feature in unique_features:
        # Set strand to "<+>" for all features except CDS
        strand = strand_map[feature.location.strand] if feature.type == 'CDS' else "<+>"

        # Handle compound locations
        if isinstance(feature.location, CompoundLocation):
            parts = feature.location.parts
        else:
            parts = [feature.location]

        # Check if the feature is circular
        is_circular = parts[0].start > parts[-1].start

        feature_sequence = ""
        for part in parts:
            start, end = part.start, part.end
            if end > start:
                feature_sequence += sequence[start:end]
            else:  # Handling circular DNA
                feature_sequence += sequence[start:] + sequence[:end]

        # Process the feature
        if feature.type == 'CDS':
            if strand == "<->":
                feature_sequence = str(Seq(feature_sequence).reverse_complement())
            amino_acids = feature.qualifiers.get('translation', [str(Seq(feature_sequence).translate())])[0]
            formatted_feature = f"{strand}{amino_acids.upper()}"
        else:
            formatted_feature = f"{strand}{feature_sequence}"

        if is_circular:
            circular_features.append(formatted_feature)
        else:
            formatted_sequence.append(formatted_feature)

    # Add any remaining sequence at the beginning
    if formatted_sequence and formatted_sequence[0].startswith("<+>"):
        first_feature_start = unique_features[0].location.start
        if first_feature_start > 0:
            formatted_sequence.insert(0, f"<+>{sequence[:first_feature_start]}")

    # Append circular features at the end
    formatted_sequence.extend(circular_features)

    return "".join(formatted_sequence)
