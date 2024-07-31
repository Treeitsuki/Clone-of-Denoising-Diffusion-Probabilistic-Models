import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from ..utils.data import *
from ..utils.image import *
from ..modules.unet import UNet, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, num_steps=1000, beta_min=1e-4, beta_max=0.02, image_size=256, device="cuda"):
        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.betas = self.create_noise_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.image_size = image_size
        self.device = device

    def create_noise_schedule(self):
        return torch.linspace(self.beta_min, self.beta_max, self.num_steps)

    def add_noise(self, x, t):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * epsilon, epsilon

    def get_random_timesteps(self, n):
        return torch.randint(low=1, high=self.num_steps, size=(n,))

    def generate_samples(self, model, n, labels, guidance_scale=3):
        logging.info(f"Generating {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.num_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if guidance_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, guidance_scale)
                alpha = self.alphas[t][:, None, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
