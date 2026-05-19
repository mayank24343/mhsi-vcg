import argparse
from PIL import Image

from models.loader import load_model
from pipeline.impl1 import Pipeline1
from pipeline.impl2 import Pipeline2
from config import LVLM_MODEL_NAME

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=int, default=1,
                        choices=[1, 2],
                        help="Which implementation to run (1 or 2)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the image")
    return parser.parse_args()


def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    # load model once
    model, tokenizer, processor = load_model(LVLM_MODEL_NAME)

    # load image
    image = Image.open(args.image).convert("RGB")

    # select pipeline
    if args.impl == 1:
        pipeline = Pipeline1(model, tokenizer, processor)
    else:
        pipeline = Pipeline2(model, tokenizer, processor)

    # run
    answer = pipeline.generate(text=args.question, image=image)

    print("\n" + "~"*50)
    print("FINAL ANSWER:")
    print(answer)
    print("~"*50)


if __name__ == "__main__":
    main()