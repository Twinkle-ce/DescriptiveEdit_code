from pathlib import Path
import sys
import os
from typing import List

from PIL import Image
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import logging

sys.path.append("./attention_bridge")
sys.path.append("./pipelines")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from attention_bridge import patch_joint_attention
from pipelines.pipeline_stable_diffusion_joint_attention_lora import (
    StableDiffusionPipelineMultiModel,
)
from utils.func import seed_everything

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# -----------------------------
# Pipeline Initialization
# -----------------------------
def initialize_joint_pipeline(
    descedit_path,
    base_model_path="../MODELS/stable-diffusion-v1-5",
    active_adapters=["xy_lora"],
    mode="full",
    ext_lora="",
):
    logger.info(f"Load model from {descedit_path}")
    pipeline = StableDiffusionPipelineMultiModel.from_pretrained(
        base_model_path,
        safety_checker=None,
        torch_dtype=torch.float32,
    ).to("cuda")

    # Joint attention patch
    patch_joint_attention.apply_patch(pipeline.unet, mode=mode)
    patch_joint_attention.initialize_joint_layers(pipeline.unet)

    # Load LoRA adapters
    for lora_name in active_adapters:
        save_dir = os.path.join(descedit_path, lora_name)
        logger.info(f"Load lora weights from {save_dir}")
        pipeline.load_lora_weights(save_dir, adapter_name=lora_name)

    # Load partial model state
    state_dict = load_file(os.path.join(descedit_path, "model.safetensors"))
    filtered_state_dict = {k: v for k, v in state_dict.items() if "conv1n" in k}
    pipeline.unet.load_state_dict(filtered_state_dict, strict=False)

    pipeline = pipeline.to(torch.float16).to("cuda")
    if ext_lora == "":
        pipeline.set_adapters(active_adapters)

    patch_joint_attention.hack_lora_forward(pipeline.unet)
    patch_joint_attention.set_patch_lora_mask(pipeline.unet, "xy_lora", [1, 1])
    if ext_lora != "":
        patch_joint_attention.set_patch_lora_mask(pipeline.unet, ext_lora, [1, 1, 1, 1])
    patch_joint_attention.set_joint_attention_mask(pipeline.unet, [0, 1, 0, 1])

    # Update scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    return pipeline


# -----------------------------
# Single Image Generation
# -----------------------------
def generate_single_image(
    pipeline: StableDiffusionPipelineMultiModel,
    image_path: str,
    prompt: List[str],
    output_path: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    joint_guidance_scale: float = 1.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    negative_prompt: List[str] = None,
):
    """Generate a single edited image using the joint attention pipeline."""
    if negative_prompt is None:
        negative_prompt = [
            "monochrome, lowres, bad anatomy, worst quality, low quality"
        ] * 2

    image = Image.open(image_path)
    original_size = image.size

    # Encode prompts
    prompt_embeds, negative_embeds = pipeline.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    generator = torch.Generator(device).manual_seed(seed)
    images = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        image=image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        joint_guidance_scale=joint_guidance_scale,
        generator=generator,
        device=device,
    ).images

    # Resize to original size and save
    images[0].resize(original_size, Image.Resampling.LANCZOS).save(output_path)
    logger.info(f"Saved generated image to {output_path}")


# -----------------------------
# Main function
# -----------------------------
def main(img_path, prompt, model_path, out_dir="./test"):
    seed_everything(42)
    os.makedirs(out_dir, exist_ok=True)
    pipeline = initialize_joint_pipeline(model_path, active_adapters=["xy_lora"])
    out_path = os.path.join(out_dir, f"test.png")
    generate_single_image(
        pipeline,
        image_path=img_path,
        prompt=prompt,
        output_path=out_path,
        guidance_scale=7.5,
        joint_guidance_scale=1.5,
        num_inference_steps=50,
    )


if __name__ == "__main__":
    main(img_path="your_path", prompt="your_prompt", model_path="model_path")
