import os
import torch
import gradio as gr
from PIL import Image

from attention_bridge import patch_joint_attention
from pipelines.pipeline_stable_diffusion_joint_attention_lora import (
    StableDiffusionPipelineMultiModel,
)
from safetensors.torch import load_file
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from utils.func import seed_everything


# -----------------------------
# 初始化 pipeline
# -----------------------------
def initialize_joint_pipeline(
    descedit_path,
    base_model_path="../MODELS/stable-diffusion-v1-5",
    active_adapters=["xy_lora"],
    mode="full",
    ext_lora="",
):
    pipeline = StableDiffusionPipelineMultiModel.from_pretrained(
        base_model_path,
        safety_checker=None,
        torch_dtype=torch.float32,
    ).to("cuda")

    patch_joint_attention.apply_patch(pipeline.unet, mode=mode)
    patch_joint_attention.initialize_joint_layers(pipeline.unet)

    for lora_name in active_adapters:
        save_dir = os.path.join(descedit_path, lora_name)
        pipeline.load_lora_weights(save_dir, adapter_name=lora_name)

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

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    return pipeline


# -----------------------------
# 推理函数
# -----------------------------
def generate_image(
    input_image,
    prompt,
    negative_prompt,
    guidance_scale,
    joint_guidance_scale,
    num_inference_steps,
    seed,
):
    seed_everything(seed)

    image = Image.open(input_image)
    original_size = image.size

    prompt_list = [prompt, " "]
    negative_list = [negative_prompt] * 2

    prompt_embeds, negative_embeds = pipe.encode_prompt(
        prompt_list,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_list,
    )

    generator = torch.Generator("cuda").manual_seed(seed)
    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        image=image,
        guidance_scale=guidance_scale,
        joint_guidance_scale=joint_guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        device="cuda",
    ).images

    return images[0].resize(original_size, Image.Resampling.LANCZOS)


# -----------------------------
# 启动 Gradio
# -----------------------------
if __name__ == "__main__":
    model_path = "/path/to/your/model"
    pipe = initialize_joint_pipeline(model_path, active_adapters=["xy_lora"])

    with gr.Blocks() as demo:
        gr.Markdown("## DescriptiveEdit")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Input Image")
                prompt = gr.Textbox(
                    label="Prompt",
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="monochrome, lowres, bad anatomy, worst quality, low quality",
                )
                guidance_scale = gr.Slider(
                    1, 15, value=7.5, step=0.1, label="Guidance Scale"
                )
                joint_guidance_scale = gr.Slider(
                    0, 5, value=1.5, step=0.1, label="Joint Guidance Scale"
                )
                num_inference_steps = gr.Slider(
                    10, 100, value=50, step=1, label="Inference Steps"
                )
                seed = gr.Number(value=42, label="Seed", precision=0)
                run_button = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Output Image")

        run_button.click(
            fn=generate_image,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                guidance_scale,
                joint_guidance_scale,
                num_inference_steps,
                seed,
            ],
            outputs=output_image,
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
