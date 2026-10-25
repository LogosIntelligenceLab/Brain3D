"""MTL high-level encoder: ridge projection plus the MindEye MLP mixer.

The diffusion prior and blurry decoder stay off. The embedding is trained
against CLIP text, then handed to UMAP.
"""

import torch
from torch import nn

from models import BrainNetwork


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(input_size, out_features) for input_size in input_sizes]
        )

    def forward(self, x, subj_idx=0):
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


class HighLevelEncoder(nn.Module):
    def __init__(self, in_dim, hidden=1024, out_dim=768, n_blocks=4, drop=0.15):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.n_blocks = n_blocks
        self.ridge = RidgeRegression([in_dim], out_features=hidden)
        self.backbone = BrainNetwork(
            h=hidden,
            in_dim=hidden,
            seq_len=1,
            n_blocks=n_blocks,
            drop=drop,
            clip_size=out_dim,
            out_dim=out_dim,
            blurry_recon=False,
            clip_scale=1,
        )

    def forward(self, voxels):
        if voxels.ndim == 2:
            voxels = voxels[:, None, :]
        hidden = self.ridge(voxels, 0)
        _backbone, clip_feat, _blur = self.backbone(hidden)
        return clip_feat.flatten(1)

    def state_for_save(self):
        return {
            "state_dict": self.state_dict(),
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "out_dim": self.out_dim,
            "n_blocks": self.n_blocks,
        }


def load_highlevel(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = HighLevelEncoder(
        in_dim=ckpt["in_dim"],
        hidden=ckpt["hidden"],
        out_dim=ckpt["out_dim"],
        n_blocks=ckpt["n_blocks"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
