#!/usr/bin/env python
"""Add ROI-specific noise to nsdgeneral betas for the disorder simulations.

The noise is written into a new hdf5. Point recon_inference.py at it with
--betas_path. Gaussian and Rician use variance (std = sqrt(variance)).
Drift rate and physiological fluctuation follow the paper defaults.
"""

import argparse
import os

import h5py
import numpy as np

from noise_function import (
    add_drift_noise,
    add_gaussian_noise,
    add_psychological_noise,
    add_rician_noise,
)
from roi_mask import load_roi_mask


def main():
    parser = argparse.ArgumentParser(description="Perturb one ROI inside nsdgeneral betas")
    parser.add_argument("--betas_path", required=True)
    parser.add_argument("--mask_path", default="brain_region_masks.hdf5")
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--roi", required=True, choices=["V1", "V2", "V3", "V4", "early_vis", "higher_vis", "nsd_general"])
    parser.add_argument("--noise", default="gaussian", choices=["gaussian", "rician", "drift", "physiological"])
    parser.add_argument("--variance", type=float, default=3.0)
    parser.add_argument("--drift_rate", type=float, default=0.1)
    parser.add_argument("--fluctuation", type=float, default=0.5)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    with h5py.File(args.betas_path, "r") as handle:
        betas = np.asarray(handle["betas"][:], dtype=np.float32)
    mask = load_roi_mask(args.mask_path, args.subj, args.roi)
    if mask is None or mask.shape[0] != betas.shape[1]:
        raise ValueError("ROI mask does not match the beta matrix.")
    piece = betas[:, mask]
    if args.noise == "gaussian":
        piece = add_gaussian_noise(piece, std_dev=float(np.sqrt(args.variance)))
    elif args.noise == "rician":
        piece = add_rician_noise(piece, sigma=float(np.sqrt(args.variance)))
    elif args.noise == "drift":
        piece = add_drift_noise(piece, drift_rate=args.drift_rate)
    else:
        piece = add_psychological_noise(piece, fluctuation_intensity=args.fluctuation)
    betas[:, mask] = piece.astype(np.float32)
    out = args.out or args.betas_path.replace(".hdf5", f"_{args.roi}_{args.noise}.hdf5")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as handle:
        handle.create_dataset("betas", data=betas, compression="gzip")
    print(f"Wrote perturbed betas to {out}")


if __name__ == "__main__":
    main()
