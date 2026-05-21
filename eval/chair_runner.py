import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from typing import List

from eval.chair import CHAIRDataset, CHAIRSample, compute_chair_metrics, CHAIRMetrics
from config import DEVICE, MAX_NEW_TOKENS


# standard CHAIR caption prompt, same across all experiments
CAPTION_PROMPT = "Please describe this image in detail."


def run_chair_eval(
    pipeline,
    samples: List[CHAIRSample],
    image_dir: str,
    alpha: float,
    save_dir: str
) -> CHAIRMetrics:
    """
    Generate a caption for each image and compute CHAIR metrics.

    Args:
        pipeline:   Pipeline1 or Pipeline2 instance
        samples:    list of CHAIRSample (no captions yet)
        image_dir:  path to COCO val images
        alpha:      current alpha value (for logging only)
        save_dir:   where to save detailed results

    Returns:
        CHAIRMetrics
    """
    for sample in tqdm(samples, desc=f" [CHAIR] alpha={alpha}"):
        image_path = os.path.join(image_dir, sample.image_file)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARNING] Not found: {image_path}")
            sample.caption = ""
            continue

        sample.caption = pipeline.generate(
            text=CAPTION_PROMPT,
            image=image,
            image_path=sample.image_file
        )

    # compute metrics
    metrics = compute_chair_metrics(samples)

    # save detailed results
    os.makedirs(save_dir, exist_ok=True)
    detail_file = os.path.join(save_dir, f"alpha_{alpha}_chair_detailed.json")

    with open(detail_file, "w") as f:
        json.dump(
            [
                {
                    "image_id":            s.image_id,
                    "image_file":          s.image_file,
                    "ground_truth":        list(s.ground_truth_objects),
                    "caption":             s.caption,
                    "mentioned_objects":   list(s.mentioned_objects or []),
                    "hallucinated_objects":list(s.hallucinated_objects or []),
                }
                for s in samples
            ],
            f,
            indent=2
        )

    print(f"  [Saved] {detail_file}")
    return metrics


def run_full_chair_evaluation(
    model,
    tokenizer,
    processor,
    pipeline_class,
    annotation_file: str,
    image_dir: str,
    alphas: List[float],
    save_dir: str = "eval_results",
    num_images: int = 500,
    impl_name: str = "impl1"
):
    """
    Full CHAIR evaluation loop across all alpha values.

    Args:
        model, tokenizer, processor: loaded model
        pipeline_class: Pipeline1 or Pipeline2
        annotation_file: COCO instances_val2014.json
        image_dir: COCO val2014 images
        alphas: list of alpha values e.g. [0.0, 0.5, 1.0]
        save_dir: root output directory
        num_images: images to evaluate (paper uses 500)
        impl_name: label for output files
    """
    # build samples once reused across all alphas
    print("\n[CHAIR] Building evaluation samples...")
    dataset = CHAIRDataset(
        annotation_file=annotation_file,
        num_images=num_images,
        min_objects=3,
        seed=42
    )
    base_samples = dataset.samples

    # precompute guidance once for all images
    all_image_files = [s.image_file for s in base_samples]

    print("\n[CHAIR] Precomputing guidance for all images...")
    precompute_pipeline = pipeline_class(model, tokenizer, processor)
    precompute_pipeline.precompute_guidance_for_dataset(
        image_paths=all_image_files,
        image_dir=image_dir
    )
    shared_cache = precompute_pipeline._guidance_cache

    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"CHAIR Evaluation — alpha = {alpha}")
        print(f"{'='*60}")

        # set alpha
        model.lm_head.alpha = alpha

        # fresh pipeline with shared cache
        pipeline = pipeline_class(model, tokenizer, processor)
        pipeline._guidance_cache = shared_cache

        # deep copy samples so captions don't bleed between alpha runs
        import copy
        samples = copy.deepcopy(base_samples)

        alpha_save_dir = os.path.join(save_dir, impl_name, f"alpha_{alpha}")
        metrics = run_chair_eval(
            pipeline=pipeline,
            samples=samples,
            image_dir=image_dir,
            alpha=alpha,
            save_dir=alpha_save_dir
        )

        all_results[alpha] = metrics

        print(f"\n  Results (alpha={alpha}):")
        print(f"  {metrics}")

    # save and print summary
    _save_chair_summary(all_results, alphas, save_dir, impl_name)

    return all_results


def _save_chair_summary(all_results, alphas, save_dir, impl_name):
    """Save summary JSON and print formatted table."""

    summary = {
    str(alpha): {
        "chair_s":            round(all_results[alpha].chair_s, 2),
        "chair_i":            round(all_results[alpha].chair_i, 2),
        "recall":             round(all_results[alpha].recall, 2),
        "total_captions":     all_results[alpha].total_captions,
        "total_sentences":    all_results[alpha].total_sentences,  
        "total_mentioned":    all_results[alpha].total_mentioned,
        "total_hallucinated": all_results[alpha].total_hallucinated,
    }
    for alpha in alphas
}

    summary_file = os.path.join(save_dir, impl_name, "chair_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[CHAIR] Summary saved to {summary_file}")

    # formatted table
    print("\n" + "="*60)
    print("CHAIR EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Alpha':<8} {'CHAIR_S':>9} {'CHAIR_I':>9} {'Recall':>9}")
    print("-"*60)
    for alpha in alphas:
        m = all_results[alpha]
        print(
            f"{alpha:<8} "
            f"{m.chair_s:>9.2f} "
            f"{m.chair_i:>9.2f} "
            f"{m.recall:>9.2f}"
        )
    print("="*60)