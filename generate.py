import argparse
import torch
import os
from tqdm import tqdm
from src.modules.unet import UNet
from src.model.diffusion import Diffusion
from src.utils.image import save_images

def generate_images(args):
    device = args.device
    model = UNet(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    labels = torch.arange(args.num_classes).long().to(device)
    n_samples_per_class = args.n_samples_per_class

    for i in tqdm(range(n_samples_per_class)):
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels, cfg_scale=args.cfg_scale)
        for j, image in enumerate(sampled_images):
            save_path = os.path.join(args.output_dir, f"class_{j}_sample_{i}.png")
            save_images(image.unsqueeze(0), save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="image_generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--n_samples_per_class", type=int, default=10)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="generated_images")
    
    args = parser.parse_args()
    generate_images(args)

if __name__ == "__main__":
    main()