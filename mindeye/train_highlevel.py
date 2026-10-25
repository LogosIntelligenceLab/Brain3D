#!/usr/bin/env python
"""Train the MTL high-level encoder against CLIP text.

BiMixCo (beta=0.15) is used for the first third of training. Later epochs
switch to the soft CLIP loss. There is no diffusion prior.

The beta file must be produced by extract_mtl.py and share the trial axis of
the MindEye betas (column 5 of behav.npy is the trial index).
"""

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader

import utils
from highlevel import HighLevelEncoder


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def caption_strings(table, coco_id):
    entry = table[int(coco_id)]
    if isinstance(entry, str):
        texts = [entry]
    else:
        texts = [str(item) for item in np.atleast_1d(entry)]
    texts = [text.strip() for text in texts if text and text.strip() and text.strip() != "nan"]
    return texts or ["an object"]


def encode_text(clip_model, texts, device):
    import clip

    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        return clip_model.encode_text(tokens).float()


def main():
    parser = argparse.ArgumentParser(description="Brain3D high-level MTL encoder")
    parser.add_argument("--data_path", default="./")
    parser.add_argument("--betas_path", default="")
    parser.add_argument("--caption_path", default="", help="COCO caption .npy indexed by NSD coco id")
    parser.add_argument("--subj", type=int, default=1, choices=list(range(1, 9)))
    parser.add_argument("--num_sessions", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--mixup_pct", type=float, default=0.33)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--model_name", default="brain3d_highlevel_subj01")
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas_path = args.betas_path or os.path.join(args.data_path, f"betas_mtl_subj0{args.subj}.hdf5")
    caption_path = args.caption_path or os.path.join(args.data_path, "COCO_73k_annots_curated.npy")
    if not os.path.isfile(betas_path):
        raise FileNotFoundError(
            f"Missing MTL betas: {betas_path}. Run extract_mtl.py on the full-brain NSD betas first."
        )
    if not os.path.isfile(caption_path):
        raise FileNotFoundError(
            f"Missing captions: {caption_path}. This is the NSD/COCO caption table indexed by coco id."
        )

    with h5py.File(betas_path, "r") as handle:
        betas = torch.tensor(handle["betas"][:], dtype=torch.float32)
    captions = np.load(caption_path, allow_pickle=True)
    in_dim = betas.shape[-1]
    print(f"MTL betas {tuple(betas.shape)} from {betas_path}")

    import clip

    clip_model, _preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    train_url = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"
    dataset = (
        wds.WebDataset(train_url, resampled=True, nodesplitter=lambda urls: urls)
        .shuffle(750, initial=1500, rng=np.random.RandomState(args.seed))
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    steps_per_epoch = max(1, (750 * args.num_sessions) // args.batch_size)

    model = HighLevelEncoder(in_dim=in_dim, hidden=args.hidden, n_blocks=args.n_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=1e-2)
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_steps, pct_start=0.05
    )
    soft_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    outdir = os.path.abspath(os.path.join("train_logs", args.model_name))
    os.makedirs(outdir, exist_ok=True)

    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        use_mix = epoch < int(args.mixup_pct * args.num_epochs)
        epoch_loss = 0.0
        seen = 0
        data_iter = iter(loader)
        for _ in range(steps_per_epoch):
            behav, _past, _future, _old = next(data_iter)
            trial = behav[:, 0, 5].long()
            coco = behav[:, 0, 0].long()
            voxel = betas[trial].to(device)
            texts = [caption_strings(captions, int(idx))[0] for idx in coco]
            target = encode_text(clip_model, texts, device)
            target = F.normalize(target, dim=-1)
            if use_mix:
                voxel, perm, mix_beta, select = utils.mixco(voxel)
            pred = F.normalize(model(voxel), dim=-1)
            if use_mix:
                loss = utils.mixco_nce(pred, target, temp=0.006, perm=perm, betas=mix_beta, select=select)
            else:
                temp = soft_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                loss = utils.soft_clip_loss(pred, target, temp=temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss)
            seen += 1
            step += 1
        print(f"epoch {epoch:03d} loss {epoch_loss / max(seen, 1):.4f} mixco={use_mix}")
        if (epoch + 1) % args.ckpt_interval == 0 or epoch + 1 == args.num_epochs:
            path = os.path.join(outdir, "last.pth" if epoch + 1 == args.num_epochs else f"epoch{epoch+1:03d}.pth")
            if epoch + 1 == args.num_epochs:
                torch.save(model.state_for_save(), os.path.join(outdir, "last.pth"))
            else:
                torch.save(model.state_for_save(), path)
            print("saved", os.path.join(outdir, "last.pth" if epoch + 1 == args.num_epochs else path))


if __name__ == "__main__":
    main()
