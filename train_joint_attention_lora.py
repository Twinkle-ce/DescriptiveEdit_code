import argparse
import json
import logging
import math
import os, sys
import random
import time
from datetime import datetime
import datasets
from omegaconf import OmegaConf
from pathlib import Path
from typing import List
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from torchvision import transforms
import transformers
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from diffusers.utils.torch_utils import is_compiled_module
from data.dataset import RefDataset, collate_fn


sys.path.append("./attention_bridge")
sys.path.append("./pipelines")
from utils.peft_utils import (
    get_peft_model_state_dict,
    set_adapters_requires_grad,
)
from attention_bridge import patch_joint_attention
from utils.util import (
    load_lora_weights,
)
from utils.func import (
    seed_everything,
    save_config,
    image_grid,
)
from pipelines.pipeline_stable_diffusion_joint_attention_lora import (
    StableDiffusionPipelineMultiModel,
)
from contextlib import nullcontext

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def init_model(config, accelerator):
    """Initialize and configure models based on the given config."""
    # Load scheduler, tokenizer and text_enoder.
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config.model.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config.model.pretrained_model_name_or_path, subfolder="text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(
        config.model.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.model.pretrained_model_name_or_path, subfolder="unet"
    )

    # move model to suitable device and dtype & freeze parameters of models to save more memory
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    for model in [text_encoder, vae, unet]:
        model.requires_grad_(False)
        model.to(accelerator.device, dtype=weight_dtype)

    patch_joint_attention.apply_patch(unet, single_dir=False, mode="full")
    patch_joint_attention.initialize_joint_layers(unet)
    unet_lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=[
            "attn1n.to_k",
            "attn1n.to_q",
            "attn1n.to_v",
            "attn1n.to_out.0",
        ],
    )
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config, adapter_name="xy_lora")
    patch_joint_attention.hack_lora_forward(unet)
    unet.set_adapters(["xy_lora"])

    for param in unet.parameters():
        param.requires_grad_(False)

    trainable_loras = ["xy_lora"]
    patch_joint_attention.set_joint_layer_requires_grad(unet, trainable_loras, True)

    # unet.enable_adapters() ### 感觉此处不需要，上面的函数已经设置了
    set_adapters_requires_grad(unet, True, trainable_loras)
    patch_joint_attention.set_patch_lora_mask(unet, "xy_lora", [1, 1])
    patch_joint_attention.set_joint_attention_mask(unet, [0, 1])

    if accelerator.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    def load_model_hook(models, input_dir):
        for model in models:
            if isinstance(model, UNet2DConditionModel):
                unet = model

        for lora_name in trainable_loras:
            lora_path = os.path.join(
                input_dir, f"{lora_name}", "pytorch_lora_weights.safetensors"
            )
            save_path_tensors = os.path.join(input_dir, "model.safetensors")
            load_lora_weights(unet, save_path_tensors, adapter_name=lora_name)

        save_path = os.path.join(input_dir, "model.pth")
        state_dict = torch.load(save_path, map_location="cpu")

        unet.load_state_dict(state_dict, strict=False)

    def save_model_hook(models, weights, output_dir):
        for model in models:
            if isinstance(model, UNet2DConditionModel):
                unet = model

        state_dict = dict()

        for name, params in unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params

        save_path = os.path.join(output_dir, "model.pth")
        torch.save(state_dict, save_path)

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    return (
        noise_scheduler,
        tokenizer,
        text_encoder,
        vae,
        unet,
    )


def log_validation(config, weight_dtype, net, accelerator, global_step):
    logger.info("Running validation... ")

    raw_unet = accelerator.unwrap_model(net)

    patch_joint_attention.set_patch_lora_mask(raw_unet, "xy_lora", [1, 1])
    patch_joint_attention.set_joint_attention_mask(raw_unet, [0, 1, 0, 1])
    patch_joint_attention.set_joint_attention(raw_unet, enable=True)

    # Create the Stable Diffusion Pipeline
    pipe = StableDiffusionPipelineMultiModel.from_pretrained(
        config.model.pretrained_model_name_or_path,
        requires_safety_checker=False,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipe.unet = raw_unet
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Use torch.Generator for reproducibility
    generator = torch.Generator(accelerator.device).manual_seed(42)
    num_samples = 4
    num_prompts = 1
    # Prepare prompts and negative prompts
    image = Image.open(config.test.image).resize((256, 256))
    prompt = config.test.prompt
    negative_prompt = config.test.negative_prompt
    prompt = [prompt] * num_prompts if not isinstance(prompt, List) else prompt
    negative_prompt = (
        [negative_prompt] * num_prompts
        if not isinstance(negative_prompt, List)
        else negative_prompt
    )

    prompt_embeds_, negative_prompt_embeds_ = pipe.encode_prompt(
        prompt,
        device=accelerator.device,
        num_images_per_prompt=num_samples,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    # Generate images
    with autocast_ctx:
        images = pipe(
            prompt_embeds=prompt_embeds_,
            negative_prompt_embeds=negative_prompt_embeds_,
            image=image,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=generator,
            device="cuda",
        ).images
    # Create and save a grid of the generated images
    grid = image_grid(images, 1, num_samples)

    patch_joint_attention.set_patch_lora_mask(raw_unet, "xy_lora", [1, 1])
    patch_joint_attention.set_joint_attention_mask(raw_unet, [0, 1])
    patch_joint_attention.set_joint_attention(raw_unet, enable=True)

    # Clean up
    del pipe
    torch.cuda.empty_cache()
    img_dir = Path(config.output.image_dir)
    out_file = Path(f"{img_dir}/{global_step:06d}.jpg")
    grid.save(out_file)
    logger.info(f"\nTest Image is saved at {out_file}")


def ensure_directories_exist(config):
    """
    确保配置中的每个目录都存在。

    参数:
    config (DotMap): 配置数据。
    """
    # 遍历需要检查的目录
    directories = [
        config.output.ckpt_dir,
        config.output.log_dir,
        config.output.image_dir,
        config.output.decode_dir,
        config.output.config_dir,
    ]
    for dir_template in directories:
        os.makedirs(dir_template, exist_ok=True)
        logger.info(f"Directory ensured: {dir_template}")


def main(config):
    seed_everything(config.training.seed)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output.proj_dir, logging_dir=config.output.log_dir
    )
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with=config.training.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
        ensure_directories_exist(config)
        save_config(config, config.output.config_dir)
        accelerator.init_trackers(f"{config.output.exp_name}")
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    (
        noise_scheduler,
        tokenizer,
        text_encoder,
        vae,
        unet,
    ) = init_model(config, accelerator)

    # dataset
    train_dataset = RefDataset(
        config.data.data_json_file,
        size=config.data.resolution,
        image_root_path=config.data.data_root_path,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.training.train_batch_size,
        num_workers=config.data.dataloader_num_workers,
    )

    if config.training.scale_lr:
        config.training.learning_rate = (
            config.training.learning_rate
            * config.training.gradient_accumulation_steps
            * config.training.train_batch_size
            * accelerator.num_processes
        )

    target_modules = [
        "attn1n.to_k",
        "attn1n.to_q",
        "attn1n.to_v",
        "attn1n.to_out.0",
    ]

    lora_param_names = []
    for name, module in unet.named_modules():
        if any(target in name for target in target_modules):
            for param_name, _ in module.named_parameters():
                lora_param_names.append(f"{name}.{param_name}")

    lora_params = []
    other_params = []
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            continue
        if name in lora_param_names:
            lora_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": config.training.learning_rate,
            },
            {
                "params": other_params,
                "lr": config.training.learning_rate,
            },
        ],
        weight_decay=config.training.weight_decay,
    )

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=1, num_training_steps=500000
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if config.training.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    if config.training.max_train_steps is None:
        config.training.max_train_steps = (
            config.training.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    if overrode_max_train_steps:
        config.training.max_train_steps = (
            config.training.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    config.training.num_train_epochs = math.ceil(
        config.training.max_train_steps / num_update_steps_per_epoch
    )
    # Train!
    total_batch_size = (
        config.training.train_batch_size
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.training.train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {config.training.max_train_steps}")
    first_epoch = 0
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if config.training.resume_from_checkpoint:
        if config.training.resume_from_checkpoint != "latest":
            path = os.path.basename(config.training.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output.ckpt_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.training.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.training.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output.ckpt_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = (
                global_step * config.training.gradient_accumulation_steps
            )
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * config.training.gradient_accumulation_steps
            )
            global_step = num_update_steps_per_epoch * first_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, config.training.max_train_steps),
        initial=global_step,  # 设置初始值为global_step
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    for epoch in range(first_epoch, config.training.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            # Skip steps until we reach the resumed step
            if (
                config.training.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % config.training.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    global_step += 1
                continue

            with accelerator.accumulate(unet):
                text_input_ids = tokenizer(
                    batch["output_captions"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                text_input_ids2 = tokenizer(
                    batch["input_captions"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                with torch.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(
                        batch["edit_images"].to(accelerator.device, dtype=vae.dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    ref_latents = vae.encode(
                        batch["ref_images"].to(accelerator.device, dtype=vae.dtype)
                    ).latent_dist.sample()
                    ref_latents = ref_latents * vae.config.scaling_factor

                    encoder_hidden_states1 = text_encoder(
                        text_input_ids.to(accelerator.device)
                    )[0]
                    encoder_hidden_states2 = text_encoder(
                        text_input_ids2.to(accelerator.device)
                    )[0]
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states1, encoder_hidden_states2]
                    )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                if config.training.noise_offset > 0.0:
                    noise += config.training.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1),
                        device=noise.device,
                    )
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = torch.cat([noisy_latents, ref_latents], dim=0)
                timesteps = torch.cat([timesteps, timesteps], dim=0)

                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                noise_pred = noise_pred[:bsz]
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                avg_loss = (
                    accelerator.gather(loss.repeat(config.training.train_batch_size))
                    .mean()
                    .item()
                )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                progress_bar.update(1)
                accelerator.log({"train_loss": avg_loss}, step=global_step)
                if accelerator.is_main_process:
                    if global_step % config.training.valid_steps == 0:
                        log_validation(
                            config=config,
                            weight_dtype=torch.bfloat16,
                            net=unet,
                            accelerator=accelerator,
                            global_step=global_step,
                        )

                    if global_step % config.training.save_steps == 0:
                        save_path = os.path.join(
                            config.output.ckpt_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        unwrapped_unet = unwrap_model(unet)
                        for lora_name in ["xy_lora"]:
                            cur_save_path = os.path.join(save_path, f"{lora_name}")
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(
                                    unwrapped_unet, adapter_name=lora_name
                                )
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=cur_save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                        logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(
                {
                    "data_time": load_data_time,
                    "lr": config.training.learning_rate,
                    "step_loss": avg_loss,
                }
            )
            if global_step >= config.training.max_train_steps:
                break
            begin = time.perf_counter()
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        for lora_name in ["xy_lora"]:
            save_dir = os.path.join(config.output.ckpt_dir, f"{lora_name}")
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet, adapter_name=lora_name)
            )

            StableDiffusionPipeline.save_lora_weights(
                save_directory=save_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

        state_dict = dict()
        for name, params in unwrapped_unet.named_parameters():
            if params.requires_grad:
                state_dict[name] = params

        save_path = os.path.join(config.output.ckpt_dir, "model.pth")
        torch.save(state_dict, save_path)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
