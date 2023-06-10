# https://colab.research.google.com/drive/1Z4WG8Wunj3HE0CERjt608ALSgSzRC9ig?usp=sharing&pli=1&authuser=2
# https://github.com/clovaai/donut
# https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be

# For Document Visual Question Answering
# The gt_parses follows the format of
# [{"question" : {question_sentence}, "answer" : {answer_candidate_1}},
# {"question" : {question_sentence}, "answer" : {answer_candidate_2}}, ...],
# for example, [{"question" : "what is the model name?", "answer" : "donut"},
# {"question" : "what is the model name?", "answer" : "document understanding transformer"}].

import requests

import json
import argparse
import gradio as gr
import torch
import numpy as np
import tqdm
import shutil

from PIL import Image

from donut import DonutModel

from datasets import load_dataset_builder, load_dataset

# Optimisations
# https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
# TODO: Make configurable
torch.backends.cudnn.benchmark = True # Initial training steps will be slower
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

torch.set_float32_matmul_precision('high')

# resolution = [2560, 1920] # ±24 GB of RAM
resolution = [1280, 960] # ±6GB of RAM

pretrained_path = "naver-clova-ix/donut-base-finetuned-docvqa"
task_name = "docvqa"

# dataset_path = "https://huggingface.co/datasets/laion/water-vit-webdataset"
# ds_builder = load_dataset_builder("water-vit-webdataset")
# dataset = load_dataset("laion/water-vit-webdataset")

ds_builder = load_dataset_builder("boomb0om/watermarks-validation")
dataset = load_dataset("boomb0om/watermarks-validation")

# Summary of Data
print(ds_builder.info.description)
print(ds_builder.info.features)
print(ds_builder.info)

print(dataset["validation"][0])
print(dataset["validation"][-1])
image = Image.open(dataset["validation"][0]["path"])

# https://huggingface.co/datasets/boomb0om/watermarks-validation/raw/main/clean/008quhpf1wmpzbzb.jpg
# https://huggingface.co/datasets/boomb0om/watermarks-validation/resolve/main/clean/008quhpf1wmpzbzb.jpg

def generate_correct_data_structure(dataset, split_name):
    lines = []
    images = []

    base_path = "https://huggingface.co/datasets/boomb0om/watermarks-validation/resolve/main"

    for item in dataset[split_name]:
        images.append(item["path"])

        line = { "question": "Is there a watermark?", "answer": item["watermark"] }
        lines.append(line)

    with open(f"./watermarks-validation-donut/{split_name}/metadata.jsonl", 'w') as f:
        for i, question in enumerate(lines):
            line = {"file_name": images[i], "ground_truth": json.dumps(question)}
            f.write(json.dumps(line) + "\n")

            # shutil.copyfile(images[i], f"./watermarks-validation-donut/{split_name}/" + images[i])

            url = f"{base_path}/{images[i]}"
            response = requests.get(url, allow_redirects=True)
            open(f"./watermarks-validation-donut/{split_name}/{images[i]}", "wb").write(r.content)
            # img = Image.open(response.content)

breakpoint()
# generate_correct_data_structure(dataset, "train")
generate_correct_data_structure(dataset, "validation")
# generate_correct_data_structure(dataset, "test")


# Will need to disable image_tensors.half() in the model code in main repo
def process_image(image_array):
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")

    dtype = torch.bfloat16

    image_array.resize((resolution[0], resolution[1]), Image.Resampling.LANCZOS)
    processed_image_array = pretrained_model.encoder.prepare_input(image_array).unsqueeze(0)
    return torch.tensor(np.array(processed_image_array)).to(dev, dtype=dtype)

if "docvqa" == task_name:
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
else:  # rvlcdip, cord, ...
    task_prompt = f"<s_{task_name}>"

pretrained_model = DonutModel.from_pretrained(pretrained_path, input_size=resolution, ignore_mismatched_sizes=True)

if torch.cuda.is_available():
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device, dtype=torch.bfloat16)

pretrained_model.eval()


