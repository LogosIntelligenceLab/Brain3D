#!/usr/bin/env python
"""Fit the UMAP bank and export per-test-image sampling weights.

The bank is CLIP text embeddings of the training-image captions. UMAP is fit
on the concatenation of training f_H and that bank, with n_neighbors=50.
Stable Diffusion is conditioned on the token embedding of the sampled caption.
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader

from highlevel import load_highlevel
from umap_weights import sampling_weights


def caption_strings(table, coco_id):
    entry = table[int(coco_id)]
    if isinstance(entry, str):
        texts = [entry]
    else:
        texts = [str(item) for item in np.atleast_1d(entry)]
    texts = [text.strip() for text in texts if text and str(text).strip() not in ("", "nan")]
    return texts or ["an object"]


def webdataset(url):
    dataset = (
        wds.WebDataset(url, resampled=False, nodesplitter=lambda urls: urls)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    return DataLoader(dataset, batch_size=64, num_workers=2)


def group_voxels(loader, betas):
    buckets = {}
    for behav, _past, _future, _old in loader:
        coco = behav[:, 0, 0].long().numpy()
        trial = behav[:, 0, 5].long().numpy()
        values = betas[trial]
        for coco_id, voxel in zip(coco, values):
            buckets.setdefault(int(coco_id), []).append(voxel)
    return buckets


def average_buckets(buckets, order):
    rows = []
    for coco_id in order:
        stacked = np.stack(buckets[coco_id], axis=0).astype(np.float32)
        rows.append(stacked.mean(axis=0))
    return np.stack(rows, axis=0)


def encode_clip_text(clip_model, texts, device):
    import clip

    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(tokens).float()
    return F.normalize(embedding, dim=-1)


def encode_sd_tokens(tokenizer, text_encoder, texts, device):
    tokens = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = text_encoder(tokens.input_ids.to(device))[0]
    return hidden.float().cpu()


def git_captions(image_hdf5, coco_ids, device, batch_size=4):
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device)
    model.eval()
    captions = []
    with h5py.File(image_hdf5, "r") as handle:
        images = handle["images"]
        for start in range(0, len(coco_ids), batch_size):
            batch_ids = coco_ids[start : start + batch_size]
            pil_images = []
            for coco_id in batch_ids:
                array = np.asarray(images[int(coco_id)])
                if array.dtype != np.uint8:
                    array = (array * 255).clip(0, 255).astype(np.uint8) if array.max() <= 1 else array.astype(np.uint8)
                if array.shape[0] == 3:
                    array = np.transpose(array, (1, 2, 0))
                pil_images.append(Image.fromarray(array).convert("RGB"))
            pixels = processor(images=pil_images, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pixel_values=pixels, max_length=20)
            captions.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return captions


def embed_fmri(model, voxels, device, batch_size=64):
    rows = []
    with torch.no_grad():
        for start in range(0, len(voxels), batch_size):
            batch = torch.tensor(voxels[start : start + batch_size], device=device)
            pred = F.normalize(model(batch), dim=-1)
            rows.append(pred.cpu())
    return torch.cat(rows, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Fit Brain3D UMAP and export SDS sampling weights")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_path", default="./")
    parser.add_argument("--betas_path", default="")
    parser.add_argument("--caption_path", default="")
    parser.add_argument("--caption_source", choices=["coco", "git"], default="coco")
    parser.add_argument("--image_hdf5", default="")
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--num_sessions", type=int, default=40)
    parser.add_argument("--n_neighbors", type=int, default=50)
    parser.add_argument("--out_dir", default="umap_cache")
    parser.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--max_test", type=int, default=0, help="Export only the first N test indices. 0 means all.")
    args = parser.parse_args()

    import clip
    import joblib
    import umap
    from transformers import AutoTokenizer, CLIPTextModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    betas_path = args.betas_path or os.path.join(args.data_path, f"betas_mtl_subj0{args.subj}.hdf5")
    caption_path = args.caption_path or os.path.join(args.data_path, "COCO_73k_annots_curated.npy")
    with h5py.File(betas_path, "r") as handle:
        betas = np.asarray(handle["betas"][:], dtype=np.float32)
    if args.caption_source == "coco":
        if not os.path.isfile(caption_path):
            raise FileNotFoundError(caption_path)
        caption_table = np.load(caption_path, allow_pickle=True)
    else:
        caption_table = None

    subj = args.subj
    if subj in (3, 6):
        num_test = 2371
    elif subj in (4, 8):
        num_test = 2188
    else:
        num_test = 3000
    train_url = f"{args.data_path}/wds/subj0{subj}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"
    test_url = f"{args.data_path}/wds/subj0{subj}/new_test/0.tar"
    train_buckets = group_voxels(webdataset(train_url), betas)
    test_loader = DataLoader(
        wds.WebDataset(test_url, resampled=False, nodesplitter=lambda urls: urls)
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav"),
        batch_size=num_test,
    )
    test_buckets = group_voxels(test_loader, betas)
    train_ids = sorted(train_buckets)
    test_ids = sorted(test_buckets)
    if args.max_test > 0:
        test_ids = test_ids[: args.max_test]
    print(f"train images {len(train_ids)} test images {len(test_ids)}")

    model = load_highlevel(args.ckpt, device=device)
    f_h = embed_fmri(model, average_buckets(train_buckets, train_ids), device).numpy()

    if args.caption_source == "git":
        image_hdf5 = args.image_hdf5 or os.path.join(args.data_path, "coco_images_224_float16.hdf5")
        train_captions = git_captions(image_hdf5, train_ids, device)
    else:
        train_captions = [caption_strings(caption_table, coco_id)[0] for coco_id in train_ids]

    clip_model, _preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()
    bank_clip = []
    for start in range(0, len(train_captions), 64):
        bank_clip.append(encode_clip_text(clip_model, train_captions[start : start + 64], device).cpu())
    bank_clip = torch.cat(bank_clip, dim=0).numpy()

    features = np.concatenate([f_h, bank_clip], axis=0)
    n_neighbors = min(args.n_neighbors, features.shape[0] - 1)
    if n_neighbors < 2:
        raise ValueError("Need at least 3 embeddings to fit UMAP.")
    reducer = umap.UMAP(n_neighbors=n_neighbors, metric="cosine", random_state=42)
    reducer.fit(features)
    bank_umap = reducer.transform(bank_clip).astype(np.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder").to(device)
    text_encoder.eval()
    sd_rows = []
    for start in range(0, len(train_captions), 16):
        sd_rows.append(encode_sd_tokens(tokenizer, text_encoder, train_captions[start : start + 16], device))
    bank_sd = torch.cat(sd_rows, dim=0).half()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"embeddings": bank_sd, "captions": train_captions}, os.path.join(args.out_dir, "bank_sd.pt"))
    np.save(os.path.join(args.out_dir, "bank_umap.npy"), bank_umap)
    joblib.dump(reducer, os.path.join(args.out_dir, "umap_reducer.joblib"))

    test_voxels = average_buckets(test_buckets, test_ids)
    test_h = embed_fmri(model, test_voxels, device)
    weight_dir = os.path.join(args.out_dir, "weights")
    os.makedirs(weight_dir, exist_ok=True)
    index_map = {}
    bank_umap_t = torch.tensor(bank_umap)
    for index, (coco_id, embedding) in enumerate(zip(test_ids, test_h)):
        coord = reducer.transform(embedding[None].numpy()).astype(np.float32)[0]
        weights = sampling_weights(torch.tensor(coord), bank_umap_t)
        torch.save(
            {"weights": weights, "index": index, "coco_id": int(coco_id)},
            os.path.join(weight_dir, f"test_{index:04d}.pt"),
        )
        index_map[str(index)] = int(coco_id)
    with open(os.path.join(args.out_dir, "index_map.json"), "w", encoding="utf-8") as handle:
        json.dump(index_map, handle)
    print(f"Wrote bank and {len(test_ids)} weight files to {args.out_dir}")


if __name__ == "__main__":
    main()
