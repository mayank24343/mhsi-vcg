import argparse
from models.loader import load_model
from pipeline.impl1 import Pipeline1
from pipeline.impl2 import Pipeline2
from eval.pope_runner import run_full_evaluation
from config import LVLM_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=int, default=1, choices=[1, 2])
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="Path to COCO instances_val2014.json")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to COCO val2014 images directory")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0],
                        help="Alpha values to evaluate")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--questions_per_image", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default="eval_results")
    return parser.parse_args()


def main():
    args = parse_args()

    # load model once — reused across all alpha values
    model, tokenizer, processor = load_model(LVLM_MODEL_NAME)

    pipeline_class = Pipeline1 if args.impl == 1 else Pipeline2
    impl_name = f"impl{args.impl}"

    run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        pipeline_class=pipeline_class,
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        alphas=args.alphas,
        save_dir=args.save_dir,
        num_images=args.num_images,
        questions_per_image=args.questions_per_image,
        impl_name=impl_name
    )


if __name__ == "__main__":
    main()