import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.ndimage import label
from transparent_background import Remover


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# remover = Remover(device='cuda', ckpt='model/pretrained/transparent_background.pth') 
remover = Remover(device='cuda', ckpt='./transparent_background.pth') 


class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)

        image = self.interface([image])[0]
        image = np.array(image)

        return image

class DPT():
    def __init__(self, task='depth', device='cuda'):

        self.task = task
        self.device = device

        from dpt import DPTDepthModel

        if task == 'depth':
            path = './omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else: # normal
            path = './omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

        # load model
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)


    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal




def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img
try:
    with open('./fine/TOKEN', 'r') as f:
        token = f.read().replace('\n', '')  # remove the last \n!
        print(f'[INFO] loaded hugging face access token from ./TOKEN!')
except FileNotFoundError as e:
    token = True
    print(
        f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./evals/final_subj01_pretrained_40sess_24bs/final_subj01_pretrained_40sess_24bs_all_enhancedrecons_', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--out_dir', default='./output', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")
    parser.add_argument('--dont_recenter', dest='recenter', action='store_false')
    opt = parser.parse_args()

    for chosen_seed in range(8):
        pt_file = torch.load(opt.base_dir+str(chosen_seed)+'.pt')
        for image_id in range(pt_file.shape[0]):
            print("image_id",image_id)
            selected_image_np = pt_file[image_id].permute(1, 2, 0).cpu().numpy() * 255
            selected_image_np = cv2.cvtColor(selected_image_np, cv2.COLOR_RGB2BGR)
            image = Image.fromarray(np.uint8(selected_image_np)).convert('RGB')
            # import pdb;pdb.set_trace()
            mask = np.array(remover.process(image, type='map'))[..., 0]
            mask = mask > mask.mean() / 2
            
            out_rgba = os.path.join(opt.out_dir, f'{image_id}_{chosen_seed}_rgba.png')
            out_depth = os.path.join(opt.out_dir, f'{image_id}_{chosen_seed}_depth.png')
            out_normal = os.path.join(opt.out_dir, f'{image_id}_{chosen_seed}_normal.png')
            out_mask = os.path.join(opt.out_dir, f'{image_id}_{chosen_seed}_mask.png')
            out_caption = os.path.join(opt.out_dir, f'{image_id}_{chosen_seed}_caption.txt')
            
            image = (selected_image_np).astype(np.uint8)
            
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # import pdb;pdb.set_trace()
            carved_image = np.concatenate((image, mask[..., np.newaxis].astype(np.uint8)*255), -1)
            # predict depth
            print(f'[INFO] depth estimation...')
            dpt_depth_model = DPT(task='depth')
            depth = dpt_depth_model(image)[0]
            depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
            depth[~mask] = 0
            depth = (depth * 255).astype(np.uint8)
            del dpt_depth_model

            # predict normal
            print(f'[INFO] normal estimation...')
            dpt_normal_model = DPT(task='normal')
            normal = dpt_normal_model(image)[0]
            normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
            normal[~mask] = 0
            del dpt_normal_model

            # recenter
            if opt.recenter:
                print(f'[INFO] recenter...')
                final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
                final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
                final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)
                final_mask = np.zeros((opt.size, opt.size), dtype=np.uint8)

                coords = np.nonzero(mask)
                x_min, x_max = coords[0].min(), coords[0].max()
                y_min, y_max = coords[1].min(), coords[1].max()
                h = x_max - x_min
                w = y_max - y_min
                desired_size = int(opt.size * (1 - opt.border_ratio))
                scale = desired_size / max(h, w)
                h2 = int(h * scale)
                w2 = int(w * scale)
                x2_min = (opt.size - h2) // 2
                x2_max = x2_min + h2
                y2_min = (opt.size - w2) // 2
                y2_max = y2_min + w2
                final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
                final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
                final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
                final_mask[x2_min:x2_max, y2_min:y2_max] = cv2.resize((mask.astype(np.uint8)*255)[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

            else:
                final_rgba = carved_image
                final_depth = depth
                final_normal = normal
                final_mask = mask.astype(np.uint8)*255

            # write output
            cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite(out_depth, final_depth)
            cv2.imwrite(out_normal, final_normal)
            cv2.imwrite(out_mask , final_mask)