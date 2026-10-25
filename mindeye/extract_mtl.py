#!/usr/bin/env python
"""Extract MTL voxels from full-brain NSD betas.

This script reads a full-brain beta volume and a mask on the same grid.

FreeSurfer labels commonly used for this mask, if you pass aparc+aseg:

    17, 53   hippocampus
    18, 54   amygdala
    1016, 2016   parahippocampal cortex

The trial axis of the output must match MindEye betas: behav column 5 indexes it.
"""

import argparse
import os

import h5py
import numpy as np


def load_nifti(path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("extract_mtl.py needs nibabel. Install it with: pip install nibabel") from exc
    image = nib.load(path)
    return np.asanyarray(image.dataobj), image.shape


def beta_volume(args):
    if args.beta_nii:
        data, _shape = load_nifti(args.beta_nii)
        if data.ndim != 4:
            raise ValueError(f"Expected a 4D nifti (X, Y, Z, trials), got {data.shape}")
        return np.transpose(data, (3, 0, 1, 2))
    if not args.beta_hdf5:
        raise ValueError("Pass --beta_nii or --beta_hdf5.")
    with h5py.File(args.beta_hdf5, "r") as handle:
        betas = np.asarray(handle[args.hdf5_key])
    if betas.ndim == 4:
        return betas
    if betas.ndim == 2:
        if not args.volume_shape:
            raise ValueError("Flat betas need --volume_shape X,Y,Z so the mask can be applied.")
        shape = tuple(int(part) for part in args.volume_shape.split(","))
        if int(np.prod(shape)) != betas.shape[1]:
            raise ValueError(f"volume_shape {shape} does not match beta width {betas.shape[1]}")
        return betas.reshape((betas.shape[0],) + shape)
    raise ValueError(f"Unsupported beta rank {betas.ndim}")


def main():
    parser = argparse.ArgumentParser(description="Extract MTL voxels for the high-level encoder")
    parser.add_argument("--beta_nii", default="")
    parser.add_argument("--beta_hdf5", default="")
    parser.add_argument("--hdf5_key", default="betas")
    parser.add_argument("--volume_shape", default="", help="X,Y,Z when betas are stored flat")
    parser.add_argument("--mask_nii", required=True)
    parser.add_argument("--labels", default="", help="Comma-separated labels to keep. Empty keeps every nonzero voxel.")
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    betas = beta_volume(args).astype(np.float32)
    mask, _shape = load_nifti(args.mask_nii)
    if mask.shape != betas.shape[1:]:
        raise ValueError(f"Mask shape {mask.shape} does not match beta volume {betas.shape[1:]}")
    if args.labels.strip():
        labels = {int(item) for item in args.labels.split(",") if item.strip()}
        keep = np.isin(mask, list(labels))
    else:
        keep = mask != 0
    if not np.any(keep):
        raise ValueError("MTL mask is empty.")
    selected = betas[:, keep]
    out = args.out or f"betas_mtl_subj0{args.subj}.hdf5"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as handle:
        handle.create_dataset("betas", data=selected, compression="gzip")
        handle.create_dataset("mask", data=keep.astype(np.bool_))
    print(f"Wrote {selected.shape} to {out}")


if __name__ == "__main__":
    main()
