def load_lora_weights(unet, lora_path, adapter_name=None):
    """
    Load LoRA weights into the UNet model.

    Args:
        unet: The UNet model instance.
        lora_path: Path to the LoRA weights file (in safetensors format).
        adapter_name: Optional name of the adapter to which weights will be loaded.

    Returns:
        None
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")

    # Load the LoRA weights
    try:
        from safetensors.torch import load_file

        lora_weights = load_file(lora_path, device="cpu")
    except ImportError:
        raise ImportError("Please install safetensors to load LoRA weights.")

    # Assign the weights to the model
    for name, param in unet.named_parameters():
        if adapter_name and adapter_name not in name:
            continue

        if name in lora_weights:
            param.data.copy_(lora_weights[name])
        else:
            raise KeyError(f"Weight {name} not found in LoRA file.")
