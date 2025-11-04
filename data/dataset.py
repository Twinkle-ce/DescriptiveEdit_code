import os
import json
import random
from typing import List, Dict
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor


class RefDataset(torch.utils.data.Dataset):
    """
    自定义数据集，加载编辑前后图像及对应的文本描述，支持数据增强(drop机制)。
    """

    def __init__(
        self,
        json_file: str,
        size: int = 512,
        t_drop_rate: float = 0.05,
        i_drop_rate: float = 0.10,
        ti_drop_rate: float = 0.15,
        image_root_path: str = "",
        mode: str = "train",
    ):
        super().__init__()

        self.size = size
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.mode = mode

        # 载入数据列表
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 图像预处理流水线
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # CLIP图像处理器（PIL Image -> tensor）
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # 加载原图和编辑图，如果加载失败则生成空白图
        try:
            ori_image = Image.open(
                os.path.join(self.image_root_path, item["ori"])
            ).convert("RGB")
        except Exception:
            ori_image = Image.new("RGB", (self.size, self.size), (255, 255, 255))

        try:
            edit_image = Image.open(
                os.path.join(self.image_root_path, item["edit"])
            ).convert("RGB")
        except Exception:
            edit_image = Image.new("RGB", (self.size, self.size), (255, 255, 255))

        # 加载对应json，读取文本描述
        json_path = os.path.join(
            self.image_root_path, item["ori"].replace("_ori.png", ".json")
        )
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        input_caption = json_data.get("input_caption", "") or ""
        output_caption = json_data.get("output_caption", "") or ""
        prompt = json_data.get("prompt", "") or ""

        # 图像预处理
        edit_image = self.transform(edit_image)
        ref_image = self.transform(ori_image)

        # CLIP image tensor
        clip_image = self.clip_image_processor(
            images=ori_image, return_tensors="pt"
        ).pixel_values

        # Drop机制（随机丢弃文本或图像信息）
        rand_num = random.random()
        if rand_num < self.t_drop_rate:
            output_caption = ""
            edit_image = ref_image
        elif rand_num < self.i_drop_rate:
            input_caption = ""
            ref_image = self.transform(
                Image.new("RGB", (self.size, self.size), (255, 255, 255))
            )
        elif rand_num < self.ti_drop_rate:
            output_caption = ""
            input_caption = ""
            ref_image = self.transform(
                Image.new("RGB", (self.size, self.size), (255, 255, 255))
            )

        return {
            "edit_image": edit_image,
            "ref_image": ref_image,
            "clip_image": clip_image.squeeze(0),  # squeeze去掉batch维度
            "output_caption": output_caption,
            "input_caption": input_caption,
            "prompt": prompt,
        }

    def __len__(self) -> int:
        return len(self.data)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义 collate 函数，用于 DataLoader 批量合并数据。

    Args:
        batch: List，batch中若干样本的列表

    Returns:
        dict，batch格式的数据字典
    """
    edit_images = torch.stack([item["edit_image"] for item in batch])
    ref_images = torch.stack([item["ref_image"] for item in batch])
    clip_images = torch.stack([item["clip_image"] for item in batch])

    output_captions = [item["output_caption"] for item in batch]
    input_captions = [item["input_caption"] for item in batch]
    prompts = [item["prompt"] for item in batch]

    return {
        "edit_images": edit_images,
        "ref_images": ref_images,
        "clip_images": clip_images,
        "output_captions": output_captions,
        "input_captions": input_captions,
        "prompts": prompts,
    }
