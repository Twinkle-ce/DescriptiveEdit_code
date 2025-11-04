import logging
import os
import json
import base64
import uuid
from tqdm import tqdm
import requests

# -----------------------------
# 配置日志
# -----------------------------
logging.basicConfig(
    filename="./generated_results/process.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------
# API 配置占位符
# -----------------------------
API_URL = "https://<YOUR_API_ENDPOINT>/chatgpt/completions"
API_HEADERS = {
    "Authorization": "Bearer <YOUR_API_KEY>",
    "Content-Type": "application/json",
}

# -----------------------------
# 数据路径
# -----------------------------
DATA_DIR = "../../../dataset/ultraedit_long/"
PROCESSED_FILE = "./data_split.json"
OUTPUT_DIR = "./generated_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PROCESSED_FILE, "r") as f:
    dataset = json.load(f)
    test_dataset = dataset["test"]

# -----------------------------
# Prompt 模板
# -----------------------------
KEY = "test_caption"
PROMPT_TEMPLATE = (
    'The image will be edited according to the instruction "{instruction}". '
    "Please provide a description of the edited image that focuses on the final visual result. "
    "The description must be clear, logical, and specific, fully aligned with the instruction. "
    "Do not mention the original image or the editing process. "
    "The description should emphasize the intended changes naturally within the overall depiction of the edited image. "
    "Ensure the text is between 50 and 150 words, and accurately conveys a coherent and realistic scene reflecting the edit."
)

# -----------------------------
# 主处理循环
# -----------------------------
for sample in tqdm(test_dataset, desc="Processing samples"):
    # 构造 json 文件路径
    ori_path = sample["ori"]
    json_file = os.path.join(
        DATA_DIR, ori_path.split("/")[0], ori_path.split("/")[1].split("_")[0] + ".json"
    )

    with open(json_file, "r") as f:
        sample_data = json.load(f)

    # 如果已经处理过则跳过
    if KEY in sample_data:
        logging.info(f"{json_file} already processed.")
        continue

    # 构造 prompt
    prompt = PROMPT_TEMPLATE.format(instruction=sample["prompt"])

    # 读取图像并转 base64
    image_path = os.path.join(DATA_DIR, ori_path)
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "prompt": prompt,
        "messages": [
            {
                "role": "user",
                "content": "data:image/png;base64," + img_base64,
                "contentType": "image",
            }
        ],
        "sessionId": str(uuid.uuid4()),
        "provider": "<YOUR_PROVIDER>",
        "model": "<YOUR_MODEL>",
    }

    # 请求接口
    try:
        response = requests.post(API_URL, headers=API_HEADERS, json=payload)
        response_data = response.json()
        generated_text = response_data["data"]["content"].replace("\n", "")
        sample_data[KEY] = generated_text

        # 保存更新后的 JSON
        with open(json_file, "w") as f:
            json.dump(sample_data, f, indent=4)

        # 记录成功
        logging.info(f"Processed: {json_file}")
        with open(os.path.join(OUTPUT_DIR, "generated.txt"), "a") as f:
            f.write(image_path + "\n")

    except Exception as e:
        logging.error(f"Failed processing {image_path}: {e}")
        with open(os.path.join(OUTPUT_DIR, "except_file.txt"), "a") as f:
            f.write(image_path + "\n")
