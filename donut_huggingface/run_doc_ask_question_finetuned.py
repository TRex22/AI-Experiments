# https://colab.research.google.com/drive/1Z4WG8Wunj3HE0CERjt608ALSgSzRC9ig?usp=sharing&pli=1&authuser=2
# https://github.com/clovaai/donut
# https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be

import argparse
import gradio as gr
import torch
import numpy as np
from PIL import Image

from donut import DonutModel

def process_image(image_array):
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")

    dtype = torch.bfloat16
    return torch.tensor(np.array(image_array)).to(dev, dtype=dtype)

def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt, task_name

    input_img = process_image(Image.fromarray(input_img))
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]

    return output

def demo_process(input_img):
    global pretrained_model, task_prompt, task_name

    input_img = process_image(Image.fromarray(input_img))
    output = pretrained_model.inference(image=input_img, prompt=task_prompt)["predictions"][0]

    return output

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="docvqa")
parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-docvqa")
args, left_argv = parser.parse_known_args()

task_name = args.task
if "docvqa" == task_name:
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
else:  # rvlcdip, cord, ...
    task_prompt = f"<s_{task_name}>"

pretrained_model = DonutModel.from_pretrained(args.pretrained_path, ignore_mismatched_sizes=True)

if torch.cuda.is_available():
   pretrained_model.half()
   device = torch.device("cuda")
   pretrained_model.to(device, dtype=torch.bfloat16)
else:
   pretrained_model.encoder.to(torch.bfloat16)

#pretrained_model.half()
#pretrained_model.encoder.to(torch.bfloat16)

pretrained_model.eval()

demo = gr.Interface(
    fn=demo_process_vqa if task_name == "docvqa" else demo_process,
    inputs=["image", "text"] if task_name == "docvqa" else "image",
    outputs="json",
    title=f"Donut 🍩 demonstration for `{task_name}` task",
)
demo.launch()
