import yaml
from dataclasses import asdict
from typing import Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoProcessor
from train import VL, DefaultTrainingArguments, HfArgumentParser
from model import Clip_FlanT5
from peft import PeftModel
import argparse
import torch
from PIL import Image
import requests
import pdb


def main():
    parser = HfArgumentParser(DefaultTrainingArguments)
    parser.add_argument("--model_checkpoint", type=str, default="./vl/cfqalt5s/checkpoints/epoch=2-step=6060.ckpt", help="model checkpoint path")
    args, inf_args = parser.parse_args_into_dataclasses()
    args.training_steps = 1
    args.use_lora = False

    # load datasets
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name_or_path, cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained(args.vision_model_name_or_path, cache_dir=args.cache_dir).image_processor
    
    vl_model = Clip_FlanT5(args)
    checkpoint = torch.load(inf_args.model_checkpoint)
    vl_model.vision_projector.load_state_dict(checkpoint["state_dict"])
    vl_model.text_decoder = PeftModel.from_pretrained(vl_model.text_decoder, checkpoint["adapter"])

    vl_model.cuda()

    urls = [
        "https://i.pinimg.com/736x/66/01/6c/66016c3ba27c0e04f39e2bd81a934e3e--anita-ekberg-bob-hope.jpg",  # author : a life in photography -- in pictures     pred: actor in a room with his wife
        "https://i.dailymail.co.uk/i/pix/2014/11/05/1415187324676_wps_31_Home_is_a_little_Deer_Ivy.jpg",   # the - bedroom stone cottage can sleep people      pred: bedroom with a view of the garden
        "https://worldjourneysdiscover.files.wordpress.com/2014/07/kyoto-07.jpg?w=860&h=645",              # party in the park under cherry blossoms           pred: a group of students in the park.
    ]
    # image = Image.open(requests.get(url, stream=True).raw)
    images = [Image.open(requests.get(url, stream=True).raw) for url in urls]

    inputs = processor(images=images, return_tensors="pt")

    # add bos token for text decoder
    inputs["input_ids"] = torch.tensor([[0]] * inputs["pixel_values"].shape[0])
    inputs["attention_mask"] = torch.tensor([[1]] * inputs["pixel_values"].shape[0])

    inputs = {k: v.cuda() for k, v in inputs.items()}

    vl_model.eval()

    outputs = vl_model.generate(**inputs)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(outputs)

if __name__ == "__main__":
    main()