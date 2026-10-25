# Brain3D: Generating 3D Objects from fMRI

### [Project page](https://brain-3d.github.io/) | [Paper](https://arxiv.org/abs/2405.15239)

> **Brain3D: Generating 3D Objects from fMRI**  
> Yuankun Yang, Li Zhang (Corresponding Author), Ziyang Xie, Zhiyuan Yuan, Jianfeng Feng, Xiatian Zhu, Yu-Gang Jiang

![Model illustration](image/figure1.jpg)

This repository is the official implementation of [Brain3D](https://arxiv.org/abs/2405.15239) (IJCV). It reconstructs a 3D object from the fMRI response to a single 2D image.

The low-level path is [MindEye2](https://github.com/MedARC-AI/MindEyeV2) (MLP, diffusion prior, unCLIP) restricted to V1–V4. The high-level path is a separate MLP on medial temporal lobe (MTL) voxels, trained against CLIP text. UMAP (`n_neighbors=50`) turns that embedding into a distribution over training-set captions. 3D generation is a [threestudio](https://github.com/threestudio-project/threestudio) Magic123 / HiFA pipeline: a NeRF stage supervised by Zero-1-to-3 and Stable Diffusion, then a DMTet stage whose high-level Score Distillation Sampling (SDS) uses a depth ControlNet. Each SDS step draws one caption from the UMAP weights.

![Pipeline](image/figure2.jpg)

MindEye2 and threestudio remain under their own licenses (`mindeye/UPSTREAM_LICENSE`, `threestudio/LICENSE`).

## Layout

```
mindeye/        fMRI encoders, UMAP, background removal
threestudio/    NeRF and DMTet generation
```

Run the fMRI commands from `mindeye/`. Run the 3D commands from `threestudio/`.

## 1. Installation

Two environments are required. Both were developed on Linux with an NVIDIA GPU.

### fMRI environment

Agree to the [NSD terms](https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions) first.

```bash
cd mindeye
bash setup.sh
source fmri/bin/activate
```

`setup.sh` installs PyTorch 2.1, the MindEye2 stack, `umap-learn`, and `nibabel`.

### 3D environment

Follow [threestudio/README.md](threestudio/README.md) through `pip install -r requirements.txt`, including ninja, CUDA, and the tiny-cuda-nn / nvdiffrast build. PyTorch 2.0+cu118 is a known-good combination. `requirements.txt` already lists `controlnet_aux`, which the depth ControlNet guidance imports.

Zero-1-to-3, Stable Diffusion 1.5, and the depth ControlNet are downloaded from Hugging Face on the first run (`bennyguo/zero123-diffusers`, `runwayml/stable-diffusion-v1-5`, `lllyasviel/control_v11f1p_sd15_depth`). Log in with `huggingface-cli login` if a download is gated.

## 2. Data

Place the MindEye2 release in `mindeye/` (or pass `--data_path`). The subset used here is the same one documented by MindEye2:

- `betas_all_subj0${SUBJ}_fp32_renorm.hdf5`
- `brain_region_masks.hdf5`
- `coco_images_224_float16.hdf5`
- `wds/subj0${SUBJ}/train/*.tar` and `wds/subj0${SUBJ}/new_test/0.tar`
- `COCO_73k_annots_curated.npy`, indexed by the NSD COCO id in `behav[:, 0, 0]`
- MindEye2 checkpoints and the unCLIP weights named in [MindEye2's README](https://github.com/MedARC-AI/MindEyeV2) (`unclip6_epoch0_step110000.ckpt`, `sd_image_var_autoenc.pth`, `bigG_to_L_epoch8.pth`)

Download them from [pscotti/mindeyev2](https://huggingface.co/datasets/pscotti/mindeyev2). Do not commit these files.

`behav.npy` column 0 is the COCO id. Column 5 is the trial index into the beta matrix. Test index `i` in every command below is position `i` in `np.unique` of the test COCO ids (sorted). `umap_cache/index_map.json` records that mapping.

### MTL betas

```bash
cd mindeye
python extract_mtl.py \
  --beta_nii /path/to/subj01_func1pt8mm_betas.nii.gz \
  --mask_nii /path/to/aparc+aseg_in_func1pt8mm.nii.gz \
  --labels 17,53,18,54,1016,2016 \
  --subj 1 \
  --out betas_mtl_subj01.hdf5
```

The nifti volume must be `(X, Y, Z, trials)` on the same grid as the mask. Flat hdf5 betas need `--beta_hdf5` and `--volume_shape X,Y,Z`. Labels `17/53`, `18/54`, and `1016/2016` are FreeSurfer hippocampus, amygdala, and parahippocampal cortex. The trial axis must be the same order as `betas_all_subj01_fp32_renorm.hdf5`.

## 3. Low-level encoder (V1–V4)

This is MindEye2 training with `--voxel_roi early_vis`, which keeps the union of V1, V2, V3, and V4. The loss is BiMixCo (`beta=0.15`) for the first third of training and soft CLIP after that, against CLIP image embeddings. The diffusion prior and unCLIP decoder stay in the model.

```bash
cd mindeye
source fmri/bin/activate
accelerate launch --mixed_precision=fp16 Train.py \
  --model_name=brain3d_lowlevel_subj01 \
  --no-multi_subject --subj=1 \
  --batch_size=24 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 \
  --use_prior --prior_scale=30 --clip_scale=1 --blurry_recon --blur_scale=.5 \
  --no-use_image_aug --n_blocks=4 --hidden_dim=4096 --num_sessions=40 \
  --voxel_roi=early_vis \
  --multisubject_ckpt=/path/to/final_multisubject_subj01
```

`--voxel_roi all` keeps the full nsdgeneral vector. Checkpoints go to `train_logs/brain3d_lowlevel_subj01/`.

Reconstruct the test set. With `--indices` empty, row `i` of the saved tensor is test index `i`.

```bash
python recon_inference.py \
  --model_name=brain3d_lowlevel_subj01 \
  --subj=1 --hidden_dim=4096 --n_blocks=4 \
  --voxel_roi=early_vis \
  --num_seeds=1 \
  --recon_dir=recons
```

Remove the background and recenter to an RGBA image for Zero-1-to-3. `carvekit` and `transparent-background` must be installed, and `transparent_background.pth` must sit next to `seg_image.py`.

```bash
python seg_image.py \
  --base_dir evals/brain3d_lowlevel_subj01/brain3d_lowlevel_subj01_all_recons_ \
  --out_dir output
```

The file consumed above is `evals/.../brain3d_lowlevel_subj01_all_recons_0.pt`. Outputs are `output/${INDEX}_0_rgba.png`.

## 4. High-level encoder (MTL)

No diffusion prior. The target is the CLIP ViT-L/14 text embedding of the COCO caption. The same BiMixCo then soft-CLIP schedule is used.

```bash
python train_highlevel.py \
  --data_path . \
  --betas_path betas_mtl_subj01.hdf5 \
  --caption_path COCO_73k_annots_curated.npy \
  --subj 1 --num_sessions 40 \
  --batch_size 32 --num_epochs 150 --mixup_pct 0.33 \
  --hidden 1024 --n_blocks 4 \
  --model_name brain3d_highlevel_subj01
```

The final checkpoint is `train_logs/brain3d_highlevel_subj01/last.pth`.

## 5. UMAP bank and sampling weights

UMAP is fit on the concatenation of training `f_H` and the CLIP text embeddings of the training captions (`n_neighbors=50`, cosine metric). For each test fMRI vector the script projects `f_H`, then writes

```
s_j = cos(U_H, U_Hj)
v_j = s_j / ||U_Hj||
w_j = v_j / sum_k v_k
```

Negative similarities are clipped to zero. `bank_sd.pt` stores the Stable Diffusion 1.5 token embeddings of those captions so SDS can consume them directly.

```bash
python fit_umap.py \
  --ckpt train_logs/brain3d_highlevel_subj01/last.pth \
  --data_path . \
  --betas_path betas_mtl_subj01.hdf5 \
  --caption_path COCO_73k_annots_curated.npy \
  --subj 1 --num_sessions 40 \
  --n_neighbors 50 \
  --out_dir umap_cache
```

Outputs:

- `umap_cache/bank_sd.pt`
- `umap_cache/weights/test_0007.pt` for test index 7
- `umap_cache/index_map.json`

## 6. Perceptual stage (NeRF)

From `threestudio/`, with that environment activated. 2000 steps is the setting used for the paper runs. The command prints the trial directory; its `ckpts/last.ckpt` is the coarse model.

```bash
cd ../threestudio
bash run_brain3d_coarse.sh 7 ../mindeye/output/7_0_rgba.png
```

Arguments are `INDEX RGBA [BANK] [WEIGHTS] [GPU]`. The default bank and weights paths are `../mindeye/umap_cache/`. Override them when the cache lives elsewhere:

```bash
bash run_brain3d_coarse.sh 7 /abs/path/7_0_rgba.png /abs/path/bank_sd.pt /abs/path/weights/test_0007.pt 0
```

Equivalent `launch.py` call:

```bash
python launch.py \
  --config configs/brain3d-coarse-sd.yaml \
  --train --gpu 0 \
  data.image_path=../mindeye/output/7_0_rgba.png \
  system.prompt_processor.bank_path=../mindeye/umap_cache/bank_sd.pt \
  system.prompt_processor.weights_path=../mindeye/umap_cache/weights/test_0007.pt \
  trainer.max_steps=2000
```

The geometry is an implicit volume. Zero-1-to-3 is conditioned on the RGBA image. Stable Diffusion 1.5 SDS is conditioned on one caption resampled from `w_j` at every step. A 360-degree video is written at the end of training. Press `ctrl+c` once to skip to that test render.

## 7. Semantic stage (DMTet + depth ControlNet)

Pass the coarse checkpoint as the third argument.

```bash
bash run_brain3d_refine.sh 7 \
  ../mindeye/output/7_0_rgba.png \
  outputs/brain3d-coarse-sd/<trial>/ckpts/last.ckpt
```

Arguments are `INDEX RGBA COARSE_CKPT [BANK] [WEIGHTS] [GPU]`.

This converts the NeRF to a tetrahedral SDF grid (DMTet, resolution 128) and keeps training for 2000 steps. Low-level guidance is still Zero-1-to-3. High-level guidance is Stable Diffusion with `lllyasviel/control_v11f1p_sd15_depth`, conditioned on the depth of the current view. The rasterizer min-max normalizes foreground depth and maps nearer points to brighter values. The text condition is again one UMAP sample per step.

Meshes and videos are under `outputs/brain3d-refine-sd/<trial>/`.

## 8. ROI perturbation

`perturb_roi.py` writes a new beta matrix. Only the chosen ROI is changed. Reconstruct it with the low-level model by pointing `--betas_path` at that file and leaving `--voxel_roi early_vis`, so the encoder still reads V1–V4 and the corrupted region stays inside that vector.

```bash
cd ../mindeye
python perturb_roi.py \
  --betas_path betas_all_subj01_fp32_renorm.hdf5 \
  --mask_path brain_region_masks.hdf5 \
  --subj 1 --roi V1 \
  --noise gaussian --variance 3.0 \
  --out betas_subj01_V1_gaussian.hdf5

python recon_inference.py \
  --model_name=brain3d_lowlevel_subj01 \
  --subj=1 --hidden_dim=4096 --n_blocks=4 \
  --voxel_roi=early_vis \
  --betas_path betas_subj01_V1_gaussian.hdf5
```

`--noise` is `gaussian`, `rician`, `drift`, or `physiological`. Defaults are variance 3.0, drift rate 0.1, and physiological fluctuation 0.5. Gaussian and Rician use `std = sqrt(variance)`. Other ROIs in `brain_region_masks.hdf5` are `V2`, `V3`, `V4`, `higher_vis`, and `nsd_general`.

## Citation

```bibtex
@article{yang2024brain3d,
  title={Brain3D: Generating 3D Objects from fMRI},
  author={Yang, Yuankun and Zhang, Li and Xie, Ziyang and Yuan, Zhiyuan and Feng, Jianfeng and Zhu, Xiatian and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2405.15239},
  year={2024}
}
```

If you use the fMRI encoder or the 3D backbone, also cite [MindEye2](https://arxiv.org/abs/2403.11207) and [threestudio](https://github.com/threestudio-project/threestudio).
