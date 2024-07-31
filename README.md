<div align="center">
<h1>
    <br>
     Clone of Denoising Diffusion Probabilistic Models
    <br>
</h1>

</div>

> [!NOTE]
> This is the Final Task of Visual Media.

# Report
### Why this paper is important (what the technical core is, why the paper is accepted)
DDPM produces high-quality images by learning noise removal processes. It has demonstrated image generation performance superior to conventional GAN and VAE.

### What you have implemented
In the original code, the model was implemented using TensorFlow. In this repository, the model was implemented using PyTorch. In addition, I implemented a program to generate images based on the weight data.

# 🖥️ Demo
```bash
python generate.py --model_path ./models/DDPM_conditional_400/ema_ckpt.pt --output_dir test
```
The following images were generated based on CelebA training dataset.

<p>
    <img src="./image/test1/class_9_sample_4.png" width="224"/>
    <img src="./image/test2/class_0_sample_2.png" width="224"/>
    <img src="./image/test2/class_0_sample_1.png" width="224"/>
    <!--<img src="./image/test2/class_9_sample_3.png" width="224"/>-->
</p>
<p>
    <img src="./image/test1/class_0_sample_0.png" width="224"/>
    <!--<img src="./image/test1/class_0_sample_1.png" width="224"/>-->
    <img src="./image/test1/class_0_sample_6.png" width="224"/>
    <img src="./image/test1/class_9_sample_1.png" width="224"/>
</p>

#  🛠️ Acknowledgement
- [Jonathan Ho, et.al, "Denoising Diffusion Probabilistic Models", NeurIPS2020](https://arxiv.org/abs/2006.11239)
- [Original GitHub](https://github.com/hojonathanho/diffusion)
- [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


# 🧾 Running code
1. Clone my repository
    ```bash
    git clone https://github.com/Treeitsuki/Clone-of-Denoising-Diffusion-Probabilistic-Models.git
    ```

1. Run ShellScript
    ```bash
    bash setup.sh
    ```

1. Train Diffusion Model
    ```bash
    python train.py
    ```

1. Generate Images
    ```bash
    python generate.py --model_path ./weight/yourckpt.pt --output_dir ./output/path
    ```
