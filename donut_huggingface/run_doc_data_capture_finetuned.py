# https://colab.research.google.com/drive/1Z4WG8Wunj3HE0CERjt608ALSgSzRC9ig?usp=sharing&pli=1&authuser=2
# https://github.com/clovaai/donut
# https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be

import argparse
import gradio as gr
import torch
import numpy as np

from PIL import Image

from donut import DonutModel

# Optimisations
# https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
# TODO: Make configurable
torch.backends.cudnn.benchmark = True # Initial training steps will be slower
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

torch.set_float32_matmul_precision('high')

# resolution = [2560, 1920] # ±24 GB of RAM
# resolution = [1280, 960] # ±6GB of RAM
resolution = [960, 960] # ±6GB of RAM

# Optimisations
# https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
# TODO: Make configurable
torch.backends.cudnn.benchmark = True # Initial training steps will be slower
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

torch.set_float32_matmul_precision('high')

# resolution = [2560, 1920] # ±24 GB of RAM
# resolution = [1280, 960] # ±6GB of RAM
resolution = [1920, 1440]

def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt, task_name

    input_img = process_image(Image.fromarray(input_img))
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(image=None, image_tensors=input_img, prompt=user_prompt)["predictions"][0]

    return output


def demo_process(input_img):
    global pretrained_model, task_prompt, task_name

    input_img = process_image(Image.fromarray(input_img))
    output = pretrained_model.inference(image=None, image_tensors=input_img, prompt=task_prompt)["predictions"][0]

    return output


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cord-v2")
parser.add_argument("--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-cord-v2")
args, left_argv = parser.parse_known_args()

task_name = args.task
if "docvqa" == task_name:
    task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
else:  # rvlcdip, cord, ...
    task_prompt = f"<s_{task_name}>"

pretrained_model = DonutModel.from_pretrained(args.pretrained_path, input_size=resolution, ignore_mismatched_sizes=True)

if torch.cuda.is_available():
    pretrained_model.half()
    device = torch.device("cuda")
    pretrained_model.to(device, dtype=torch.bfloat16)

pretrained_model.eval()

demo = gr.Interface(
    fn=demo_process_vqa if task_name == "docvqa" else demo_process,
    inputs=["image", "text"] if task_name == "docvqa" else "image",
    outputs="json",
    title=f"Donut 🍩 demonstration for `{task_name}` task",
)
demo.launch()
