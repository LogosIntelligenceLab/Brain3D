"""ROI masks over MindEye nsdgeneral betas.

`brain_region_masks.hdf5` stores one boolean vector per ROI, aligned with the
last dimension of `betas_all_subj0*_fp32_renorm.hdf5`. The low-level encoder
uses the union of V1, V2, V3 and V4.
"""

import os

import h5py
import numpy as np

EARLY_VISUAL = ("V1", "V2", "V3", "V4")
KNOWN_ROIS = EARLY_VISUAL + ("early_vis", "higher_vis", "nsd_general", "all")


def subject_key(subj):
    return f"subj0{int(subj)}"


def load_roi_mask(mask_path, subj, roi):
    """Return a boolean mask, or None when every voxel should be kept."""
    if roi in (None, "", "all"):
        return None
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"ROI mask file not found: {mask_path}. "
            "Download brain_region_masks.hdf5 with the MindEye2 NSD release."
        )
    names = EARLY_VISUAL if roi == "early_vis" else (roi,)
    with h5py.File(mask_path, "r") as handle:
        group = handle[subject_key(subj)]
        missing = [name for name in names if name not in group]
        if missing:
            raise KeyError(
                f"ROI {missing} not in {mask_path}:{subject_key(subj)}. "
                f"Available keys: {list(group.keys())}"
            )
        mask = None
        for name in names:
            piece = np.asarray(group[name][:], dtype=bool)
            mask = piece if mask is None else np.logical_or(mask, piece)
    if mask is None or not np.any(mask):
        raise ValueError(f"ROI {roi} for subject {subj} is empty.")
    return mask


def apply_beta_roi(betas, subj, roi, mask_path):
    """Keep voxels inside `roi`. `betas` is (trials, voxels) or a torch tensor."""
    mask = load_roi_mask(mask_path, subj, roi)
    if mask is None:
        return betas
    if hasattr(betas, "shape") and not isinstance(betas, np.ndarray):
        import torch

        if isinstance(betas, torch.Tensor):
            if mask.shape[0] != betas.shape[-1]:
                raise ValueError(
                    f"ROI mask length {mask.shape[0]} does not match "
                    f"betas last dim {betas.shape[-1]}."
                )
            return betas[..., torch.as_tensor(mask)]
    betas_np = np.asarray(betas)
    if mask.shape[0] != betas_np.shape[-1]:
        raise ValueError(
            f"ROI mask length {mask.shape[0]} does not match "
            f"betas last dim {betas_np.shape[-1]}."
        )
    return betas_np[..., mask]
