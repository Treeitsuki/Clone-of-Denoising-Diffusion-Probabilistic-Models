import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from src.utils.data import *
from src.utils.image import *
from src.modules.unet import UNet, EMA
from src.model.diffusion import Diffusion
import logging
from torch.utils.tensorboard import SummaryWriter

def train_model(config):
    setup_logging(config.experiment_name)
    device = config.device
    data_loader = load_dataset(config)
    model = UNet(num_classes=config.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    diffusion = Diffusion(image_size=config.image_size, device=device)
    tb_writer = SummaryWriter(os.path.join("logs", config.experiment_name))
    total_batches = len(data_loader)
    moving_avg = EMA(0.995)
    avg_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(config.num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        progress_bar = tqdm(data_loader)
        for batch, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.get_random_timesteps(images.shape[0]).to(device)
            noisy_images, noise = diffusion.add_noise(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(noisy_images, t, labels)
            loss = loss_fn(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            moving_avg.step_ema(avg_model, model)

            progress_bar.set_postfix(MSE=loss.item())
            tb_writer.add_scalar("MSE", loss.item(), global_step=epoch * total_batches + batch)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            generated_images = diffusion.generate_samples(model, n=len(labels), labels=labels)
            avg_generated_images = diffusion.generate_samples(avg_model, n=len(labels), labels=labels)
            plot_images(generated_images)
            save_images(generated_images, os.path.join("output", config.experiment_name, f"{epoch}.jpg"))
            save_images(avg_generated_images, os.path.join("output", config.experiment_name, f"{epoch}_avg.jpg"))
            torch.save(model.state_dict(), os.path.join("checkpoints", config.experiment_name, f"model.pt"))
            torch.save(avg_model.state_dict(), os.path.join("checkpoints", config.experiment_name, f"avg_model.pt"))
            torch.save(optimizer.state_dict(), os.path.join("checkpoints", config.experiment_name, f"optimizer.pt"))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.experiment_name = "Result of DDPM"
    args.num_epochs = 300
    args.batch_size = 14
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"./dataset"
    args.device = "cuda"
    args.learning_rate = 3e-4
    train_model(args)


if __name__ == '__main__':
    main()