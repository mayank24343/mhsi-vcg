import argparse
from models.loader import load_model
from pipeline.impl1 import Pipeline1
from pipeline.impl2 import Pipeline2
from pipeline.impl3 import Pipeline3
from eval.external_pope_runner import run_full_external_evaluation
from config import LVLM_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--random_file",     type=str, required=True,
                        help="Path to coco_pope_random.json", default = "C:\Users\Mayank\OneDrive\Desktop\MHSI\data\pope\coco_pope_random.json")
    parser.add_argument("--popular_file",    type=str, required=True,
                        help="Path to coco_pope_popular.json", default = "C:\Users\Mayank\OneDrive\Desktop\MHSI\data\pope\coco_pope_popular.json")
    parser.add_argument("--adversarial_file",type=str, required=True,
                        help="Path to coco_pope_adversarial.json", default = "C:\Users\Mayank\OneDrive\Desktop\MHSI\data\pope\coco_pope_adversarial.json")
    parser.add_argument("--detr_file",       type=str, required=True,
                        help="Path to coco_detr.json", default = "C:\Users\Mayank\OneDrive\Desktop\MHSI\data\guidance\coco_detr.json")
    parser.add_argument("--ram_file",        type=str, required=True,
                        help="Path to coco_ram.json", default = "C:\Users\Mayank\OneDrive\Desktop\MHSI\data\guidance\coco_ram.json")
    parser.add_argument("--image_dir",       type=str, required=True,
                        help="Path to COCO val2014 images", default = "C:\Users\Mayank\OneDrive\Desktop\MHSI\downloaded_images")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0])
    parser.add_argument("--save_dir", type=str, default="eval_results")
    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer, processor = load_model(LVLM_MODEL_NAME)
    pipeline_class = Pipeline3

    run_full_external_evaluation(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        pipeline_class=pipeline_class,
        question_files={
            "random":      args.random_file,
            "popular":     args.popular_file,
            "adversarial": args.adversarial_file,
        },
        detr_file=args.detr_file,
        ram_file=args.ram_file,
        image_dir=args.image_dir,
        alphas=args.alphas,
        save_dir=args.save_dir,
        impl_name=f"impl{args.impl}"
    )


if __name__ == "__main__":
    main()