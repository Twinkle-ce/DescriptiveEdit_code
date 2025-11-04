from datetime import datetime
import os
import random
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import yaml
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Union, Tuple


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def draw_multiline_text(image, text, position, max_width):
    # 加载本地字体
    font = ImageFont.truetype("../fonts/times.ttf", 15)
    draw = ImageDraw.Draw(image)

    # 拆分文本为单词列表
    words = text.split()
    lines = []
    current_line = words[0]

    # 构建每行文本，确保不超过最大宽度
    for word in words[1:]:
        # 计算当前行加上新单词后的宽度
        line_width = draw.textlength(current_line + " " + word, font=font)
        if line_width <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    # 在图像上绘制文本
    y_offset = position[1]
    for line in lines:
        draw.text((position[0], y_offset), line, font=font, fill=(0, 0, 0))
        # 获取文本高度，并增加行间距
        _, _, _, line_height = draw.textbbox((0, 0), line, font=font)
        y_offset += line_height + 1


def save_config(config, config_dir):
    # 获取当前时间，作为文件夹的一部分
    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    config_file_path = os.path.join(config_dir, f"{timestamp}.yaml")
    # 保存配置到文件
    with open(config_file_path, "w") as f:
        config_dict = OmegaConf.to_container(config, resolve=True)
        yaml.dump(config_dict, f, default_flow_style=False)


# decode latents into image with vae and save image
def recover_image(latents, i, vae, name):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    img = image[i]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60, -5, 15, -50),
        (60, 10, -5, -35),
    )

    weights_tensor = torch.t(
        torch.tensor(weights, dtype=latents.dtype).to(latents.device)
    )
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(
        latents.device
    )
    rgb_tensor = torch.einsum(
        "...lxy,lr -> ...rxy", latents, weights_tensor
    ) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)


def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    image = latents_to_rgb(latents[0])
    image.save(f"{step}.png")

    return callback_kwargs


# reconstruct image from noisy latents with timesteps
def reconstruct_image(latents, i, noise, timesteps, scheduler, vae, name, mode):
    alphas_t = scheduler.alphas_cumprod[timesteps]
    alphas_t_pre = scheduler.alphas_cumprod[timesteps - 1]
    one_minus_alphas_t = 1 - alphas_t
    one_minus_alphas_t_pre = 1 - alphas_t_pre
    sqrt_alphas_t_pre = torch.sqrt(alphas_t_pre)
    sqrt_alphas_t = torch.sqrt(alphas_t)
    sqrt_one_minus_alphas_t = torch.sqrt(one_minus_alphas_t)
    sqrt_one_minus_alphas_t_pre = torch.sqrt(one_minus_alphas_t_pre)

    if mode.lower() == "ddim":
        latent = sqrt_alphas_t_pre / sqrt_alphas_t * (
            latents - sqrt_one_minus_alphas_t * noise.to(dtype=torch.float16)
        ) + sqrt_one_minus_alphas_t_pre * noise.to(dtype=torch.float16)
    elif mode.lower() == "ddpm":
        latent = (
            latents - sqrt_one_minus_alphas_t * noise.to(dtype=torch.float16)
        ) / sqrt_alphas_t

    return recover_image(latent, i, vae, name)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def plot_images_grid(images, titles, rows=2, cols=3, save_path=None):
    """
    将多张图片展示在一个网格中，并添加标题。
    Args:
        images (list of np.ndarray or PIL.Image.Image): 图片列表。
        titles (list of str): 每张图片的标题列表。
        rows (int): 网格的行数。
        cols (int): 网格的列数。
        save_path (str, optional): 如果提供，将保存图像到文件。
    """
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for ax, img, title in zip(axes.flatten(), images, titles):
        if isinstance(img, Image.Image):
            img = np.array(img)
        ### 二维图像，可视化分析，彩色显示
        if img.ndim == 2:
            cax = ax.imshow(img, cmap="viridis")
            fig.colorbar(cax, ax=ax, orientation="vertical")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 隐藏多余的子图
    for ax in axes.flatten()[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Grid image saved to {save_path}")
    plt.close(fig)


def visualize_two_images_with_text(
    ori_image, gen_image, sample, text, combine_image_path
):
    # 创建画布和子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原始图像
    axes[0].imshow(ori_image)
    original_title = f"Original Image\n{sample['input_caption']}"
    wrapped_caption1 = "\n".join(textwrap.wrap(original_title, 65))
    axes[0].set_title(wrapped_caption1, fontsize=10, ha="center")
    axes[0].axis("off")

    # 显示编辑后的图像
    axes[1].imshow(gen_image)
    edited_title = f"Edited Image\n{sample['output_caption']}"
    wrapped_caption2 = "\n".join(textwrap.wrap(edited_title, 70))
    axes[1].set_title(wrapped_caption2, fontsize=10, ha="center")
    axes[1].axis("off")
    # 添加文字描述至画布底部
    plt.figtext(0.5, 0.02, text, wrap=True, horizontalalignment="center", fontsize=10)
    # 保存图表到指定路径并关闭画布以释放资源
    plt.savefig(combine_image_path)
    plt.close(fig)


def text_under_image(
    image: Union[np.ndarray, Image.Image],
    text: str,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_size: int = 20,
    offset: float = 0.2,
) -> Image.Image:
    # 如果输入是 numpy.ndarray，转换为 PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    offset = int(h * offset)  # 文字区域高度
    new_img = Image.new("RGB", (w, h + offset), color=(255, 255, 255))
    new_img.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    font = ImageFont.load_default()

    # 计算文本尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]  # right - left
    text_height = bbox[3] - bbox[1]  # bottom - top
    # 计算文本位置（居中对齐）
    text_x = (w - text_width) // 2
    text_y = h + (offset - text_height) // 2

    # 绘制文本
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    return new_img


def save_grid_images(images, out_path, num_rows=1, offset_ratio=0.02):
    if isinstance(images, list):
        num_images = len(images)
    elif images.ndim == 4:
        num_images = images.shape[0]
        images = [Image.fromarray(image.astype(np.uint8)) for image in images]
    else:
        images = [Image.fromarray(images.astype(np.uint8))]
        num_images = 1

    width, height = images[0].size
    num_empty = (num_rows - (num_images % num_rows)) % num_rows
    empty_image = Image.new("RGB", (width, height), color=(255, 255, 255))
    images.extend([empty_image] * num_empty)

    num_cols = (num_images + num_empty) // num_rows
    offset = int(height * offset_ratio)
    bg_width = width * num_cols + offset * (num_cols - 1)
    bg_height = height * num_rows + offset * (num_rows - 1)

    background = Image.new("RGB", (bg_width, bg_height), color=(255, 255, 255))

    for i in range(num_rows):
        for j in range(num_cols):
            img_index = i * num_cols + j
            if img_index < len(images):
                top_left_x = j * (width + offset)
                top_left_y = i * (height + offset)
                background.paste(images[img_index], (top_left_x, top_left_y))
    background.save(out_path)
    print(f"Images saved in {out_path}")
