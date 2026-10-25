#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from PIL import Image

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)


# In[5]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "final_subj01_pretrained_40sess_24bs"
    print("model_name:", model_name)

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path=./ \
                    --cache_dir=./ \
                    --model_name={model_name} --subj=1 \
                    --hidden_dim=4096 --n_blocks=4"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[6]:


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="final_subj01_pretrained_40sess_24bs",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",type=str2bool,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=4096,
)
parser.add_argument(
    "--new_test",type=str2bool,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--num_seeds", type=int, default=1,
    help="How many reconstruction samples to draw.",
)
parser.add_argument(
    "--indices", type=str, default="",
    help="Comma-separated positions in np.unique(test coco ids). Empty runs all of them.",
)
parser.add_argument(
    "--voxel_roi", type=str, default="early_vis",
    choices=["early_vis", "V1", "V2", "V3", "V4", "higher_vis", "nsd_general", "all"],
)
parser.add_argument(
    "--roi_mask_path", type=str, default="",
)
parser.add_argument(
    "--betas_path", type=str, default="",
    help="Optional replacement for betas_all_subj0*_fp32_renorm.hdf5, including perturb_roi.py outputs.",
)
parser.add_argument(
    "--recon_dir", type=str, default="recons",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)



# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}",exist_ok=True)


# In[7]:


voxels = {}
# Load hdf5 data for betas
beta_file = betas_path or f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5'
f = h5py.File(beta_file, 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
from roi_mask import apply_beta_roi
mask_path = roi_mask_path or f"{data_path}/brain_region_masks.hdf5"
betas = apply_beta_roi(betas, subj, voxel_roi, mask_path)
print(f"voxel_roi={voxel_roi} kept {betas.shape[-1]} voxels from {beta_file}")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
    
print(test_url)
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")


# In[8]:


# Prep images but don't load them all to memory
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))


# In[9]:


clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out
        
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.vae import Decoder
from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, 
                        clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt
tag='last'
outdir = os.path.abspath(f'../train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")


# In[ ]:


# setup text caption networks
from transformers import AutoProcessor, AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024

class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x
        
clip_convert = CLIPConverter()
state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device) # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
del state_dict


# In[ ]:


# prep unCLIP
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                    denoiser_config=denoiser_config,
                    first_stage_config=first_stage_config,
                    conditioner_config=conditioner_config,
                    sampler_config=sampler_config,
                    scale_factor=scale_factor,
                    disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)

ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

if indices.strip():
    index_list = [int(item) for item in indices.split(",") if item.strip()]
else:
    index_list = list(range(len(np.unique(test_images_idx))))
os.makedirs(recon_dir, exist_ok=True)
for seed in range(num_seeds):
    utils.seed_everything(seed)
    for noise_index in range(1):
        noise = 0
        region = voxel_roi
        # seed all random functions
        batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
            "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(device)}
        out = diffusion_engine.conditioner(batch)
        vector_suffix = out["vector"].to(device)
        print("vector_suffix", vector_suffix.shape)


        # In[ ]:


        # get all reconstructions
        model.to(device)
        model.eval().requires_grad_(False)

        # all_images = None
        all_blurryrecons = None
        all_recons = None
        all_predcaptions = []
        all_clipvoxels = None

        minibatch_size = 1
        num_samples_per_image = 1
        assert num_samples_per_image == 1

        if utils.is_interactive(): plotting=True

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tqdm(range(0,len(np.unique(test_images_idx)),minibatch_size)):
                if batch not in index_list:
                    continue
                uniq_imgs = np.unique(test_images_idx)[batch:batch+minibatch_size]
                voxel = None
                for uniq_img in uniq_imgs:
                    locs = np.where(test_images_idx==uniq_img)[0]
                    if len(locs)==1:
                        locs = locs.repeat(3)
                    elif len(locs)==2:
                        locs = locs.repeat(2)[:3]
                    assert len(locs)==3
                    if voxel is None:
                        voxel = test_voxels[None,locs] # 1, num_image_repetitions, num_voxels
                    else:
                        voxel = torch.vstack((voxel, test_voxels[None,locs]))
                voxel = voxel.to(device)
                for rep in range(3):
                    voxel_ridge = model.ridge(voxel[:,[rep]],0) # 0th index of subj_list
                    backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
                    if rep==0:
                        clip_voxels = clip_voxels0
                        backbone = backbone0
                        blurry_image_enc = blurry_image_enc0[0]
                    else:
                        clip_voxels += clip_voxels0
                        backbone += backbone0
                        blurry_image_enc += blurry_image_enc0[0]
                clip_voxels /= 3
                backbone /= 3
                blurry_image_enc /= 3
                        
                # Save retrieval submodule outputs
                if all_clipvoxels is None:
                    all_clipvoxels = clip_voxels.cpu()
                else:
                    all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))
                
                # Feed voxels through OpenCLIP-bigG diffusion prior
                prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                                text_cond = dict(text_embed = backbone), 
                                cond_scale = 1., timesteps = 20)
                
                pred_caption_emb = clip_convert(prior_out)
                generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
                all_predcaptions = np.hstack((all_predcaptions, generated_caption))
                print(generated_caption)
                
                # Feed diffusion prior outputs through unCLIP
                for i in range(len(voxel)):
                    samples = utils.unclip_recon(prior_out[[i]],
                                    diffusion_engine,
                                    vector_suffix,
                                    num_samples=num_samples_per_image)
                    if all_recons is None:
                        all_recons = samples.cpu()
                    else:
                        all_recons = torch.vstack((all_recons, samples.cpu()))
                    # if plotting:
                    #     for s in range(num_samples_per_image):
                    #         plt.figure(figsize=(2,2))
                    #         plt.imshow(transforms.ToPILImage()(samples[s]))
                    #         plt.axis('off')
                    #         plt.show()
                save_dir = os.path.join(recon_dir, f'{batch}_{seed}.png')
                Image.fromarray((samples.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).save(save_dir)
                if blurry_recon:
                    blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample/ 2 + 0.5).clamp(0,1)
                    
                    for i in range(len(voxel)):
                        im = torch.Tensor(blurred_image[i])
                        if all_blurryrecons is None:
                            all_blurryrecons = im[None].cpu()
                        else:
                            all_blurryrecons = torch.vstack((all_blurryrecons, im[None].cpu()))
                        # if plotting:
                        #     plt.figure(figsize=(2,2))
                        #     plt.imshow(transforms.ToPILImage()(im))
                        #     plt.axis('off')
                        #     plt.show()

                # if plotting: 
                #     print(model_name)
                #     err # dont actually want to run the whole thing with plotting=True
        # resize outputs before saving
        imsize = 256
        all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
        if blurry_recon: 
            all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()
                
        # saving
        print(all_recons.shape)
    # # You can find the all_images file on huggingface: https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals
    # torch.save(all_images,"evals/all_images.pt") 
    if blurry_recon:
        torch.save(all_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons_{seed}.pt")
    torch.save(all_recons,f"evals/{model_name}/{model_name}_all_recons_{seed}.pt")
    torch.save(all_predcaptions,f"evals/{model_name}/{model_name}_all_predcaptions_{seed}.pt")
    torch.save(all_clipvoxels,f"evals/{model_name}/{model_name}_all_clipvoxels_{seed}.pt")
    print(f"saved {model_name} outputs!")

# if not utils.is_interactive():
#     sys.exit(0)

