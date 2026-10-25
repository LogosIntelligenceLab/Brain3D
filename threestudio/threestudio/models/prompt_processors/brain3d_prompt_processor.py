from dataclasses import dataclass

import torch

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import (
    StableDiffusionPromptProcessor,
)
from threestudio.utils.typing import *


@threestudio.register("brain3d-prompt-processor")
class Brain3DPromptProcessor(StableDiffusionPromptProcessor):
    """Sample one caption token embedding at every SDS step.

    `bank_path` is the file written by mindeye/fit_umap.py (`bank_sd.pt`).
    `weights_path` is one test index (`weights/test_XXXX.pt`). Weights already
    follow the UMAP cosine normalization, so this class only draws an index.
    """

    @dataclass
    class Config(StableDiffusionPromptProcessor.Config):
        bank_path: str = ""
        weights_path: str = ""

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if not self.cfg.bank_path or not self.cfg.weights_path:
            raise ValueError(
                "brain3d-prompt-processor needs system.prompt_processor.bank_path "
                "and system.prompt_processor.weights_path."
            )
        bank = torch.load(self.cfg.bank_path, map_location="cpu")
        if isinstance(bank, dict):
            self.captions = list(bank.get("captions") or [])
            embeddings = bank["embeddings"]
        else:
            self.captions = []
            embeddings = bank
        self.bank = embeddings.float()
        loaded = torch.load(self.cfg.weights_path, map_location="cpu")
        weights = loaded["weights"] if isinstance(loaded, dict) else loaded
        weights = torch.as_tensor(weights, dtype=torch.float32).flatten()
        if weights.numel() != self.bank.shape[0]:
            raise ValueError(
                f"Weight length {weights.numel()} does not match bank {self.bank.shape[0]}."
            )
        weights = weights.clamp(min=0)
        total = float(weights.sum())
        if total <= 0:
            weights = torch.full_like(weights, 1.0 / weights.numel())
        else:
            weights = weights / total
        self.sample_weights = weights
        threestudio.info(
            f"Brain3D prompt bank: {tuple(self.bank.shape)} from {self.cfg.bank_path}"
        )

    def __call__(self) -> PromptProcessorOutput:
        index = int(torch.multinomial(self.sample_weights, 1).item())
        embedding = self.bank[index].to(self.device)
        self.text_embeddings = embedding[None]
        self.text_embeddings_vd = embedding.unsqueeze(0).repeat(len(self.directions), 1, 1)
        if index < len(self.captions):
            self.prompt = self.captions[index]
            self.prompts_vd = [prompt_fn(self.prompt) for prompt_fn in (
                direction.prompt for direction in self.directions
            )]
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            prompt=self.prompt,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            prompts_vd=self.prompts_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )
