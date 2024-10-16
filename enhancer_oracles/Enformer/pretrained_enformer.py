import torch
from enformer_pytorch import Enformer

# make sure to run pip install enformer-pytorch

# load pretrained model
model = Enformer.from_hparams(
    dim=1536,
    depth=11,
    heads=8,
    output_heads=dict(human=5313, mouse=1643),
    target_length=896,
)

# TODO: allow generated sequences to be inputted here instead of random sequences
seq = torch.randint(0, 5, (1, 196_608))  # for ACGTN, in that order (-1 for padding)
output = model(seq)

human_expr_pred = output['human']  # (1, 896, 5313)
mouse_expr_pred = output['mouse']  # (1, 896, 1643)
