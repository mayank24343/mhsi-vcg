import os
import json
import torch
from PIL import Image
from typing import List
from tqdm import tqdm

from eval.pope import POPEDataset, POPEQuestion
from eval.pope_metrics import compute_metrics, POPEMetrics
from config import DEVICE, MAX_NEW_TOKENS


def extract_yes_no(raw_answer: str) -> str:
    """
    Parse model output to extract yes/no.
    Models don't always output a clean single word.
    """
    answer = raw_answer.strip().lower()

    # check first word
    first_word = answer.split()[0] if answer.split() else ""
    if first_word in ("yes", "no"):
        return first_word

    # check if yes/no appears anywhere
    if "yes" in answer:
        return "yes"
    if "no" in answer:
        return "no"

    # default — treat as no (conservative)
    return "no"


def run_pope_eval(
    pipeline,
    questions: List[POPEQuestion],
    image_dir: str,
    sampling_name: str,
    alpha: float,
    save_dir: str
) -> POPEMetrics:
    """
    Run POPE evaluation for a given list of questions.

    Args:
        pipeline:      impl1 or impl2 pipeline instance
        questions:     list of POPEQuestion
        image_dir:     path to COCO val images
        sampling_name: "random", "popular", or "adversarial"
        alpha:         current alpha value (for logging)
        save_dir:      where to save detailed results

    Returns:
        POPEMetrics
    """
    predictions = []
    ground_truths = []
    detailed_results = []

    for q in tqdm(questions, desc=f"  [{sampling_name}] alpha={alpha}"):
        image_path = os.path.join(image_dir, q.image_file)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARNING] Image not found: {image_path}, skipping.")
            continue

        # run pipeline
        raw_answer = pipeline.generate(
            text=q.question,
            image=image,
            image_path=q.image_file
        )

        pred = extract_yes_no(raw_answer)
        predictions.append(pred)
        ground_truths.append(q.answer)

        detailed_results.append({
            "image_id": q.image_id,
            "image_file": q.image_file,
            "object": q.object_name,
            "question": q.question,
            "ground_truth": q.answer,
            "raw_answer": raw_answer,
            "prediction": pred,
            "correct": pred == q.answer
        })

    # compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    # save detailed results
    os.makedirs(save_dir, exist_ok=True)
    result_file = os.path.join(
        save_dir,
        f"alpha_{alpha}_{sampling_name}_detailed.json"
    )
    with open(result_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"  [Saved] Detailed results → {result_file}")
    return metrics


def run_full_evaluation(
    model,
    tokenizer,
    processor,
    pipeline_class,
    annotation_file: str,
    image_dir: str,
    alphas: List[float],
    save_dir: str = "eval_results",
    num_images: int = 500,
    questions_per_image: int = 6,
    impl_name: str = "impl1"
):
    """
    Full evaluation loop:
    - For each alpha
      - For each sampling strategy (random, popular, adversarial)
        - Run POPE and collect metrics

    Args:
        model, tokenizer, processor: loaded model components
        pipeline_class: Pipeline1 or Pipeline2 class
        annotation_file: COCO annotation JSON path
        image_dir: COCO val images directory
        alphas: list of alpha values to evaluate e.g. [0.0, 0.5, 1.0]
        save_dir: where to save all results
        num_images: number of images (paper uses 500)
        questions_per_image: l in the paper (paper uses 6)
        impl_name: "impl1" or "impl2" for labeling output files
    """
    samplings = ["random", "popular", "adversarial"]

    # build questions once — reuse across all alpha values
    print("\n[Eval] Building POPE questions...")
    pope = POPEDataset(
        annotation_file=annotation_file,
        num_images=num_images,
        questions_per_image=questions_per_image,
        min_objects=3,
        seed=42
    )

    # precompute all question sets
    question_sets = {
        s: pope.build(s) for s in samplings
    }

    # storage for all results
    # all_results[alpha][sampling] = POPEMetrics
    all_results = {}

    all_image_files = list(set(
    q.image_file
    for questions in question_sets.values()
    for q in questions
    ))

    # precompute guidance once before the alpha loop
    # this means detection + SVD runs once per image total
    # not once per image per alpha
    print("\n[Eval] Precomputing guidance for all images...")
    pipeline_for_precompute = pipeline_class(model, tokenizer, processor)
    pipeline_for_precompute.precompute_guidance_for_dataset(
        image_paths=all_image_files,
        image_dir=image_dir
    )
    shared_cache = pipeline_for_precompute._guidance_cache

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Evaluating alpha = {alpha}")
        print(f"{'='*60}")

        # set alpha on the lm_head
        model.lm_head.alpha = alpha

        # instantiate pipeline with current model state
        pipeline = pipeline_class(model, tokenizer, processor)

        alpha_save_dir = os.path.join(save_dir, impl_name, f"alpha_{alpha}")
        all_results[alpha] = {}

        for sampling in samplings:
            print(f"\n  Sampling strategy: {sampling}")
            questions = question_sets[sampling]

            metrics = run_pope_eval(
                pipeline=pipeline,
                questions=questions,
                image_dir=image_dir,
                sampling_name=sampling,
                alpha=alpha,
                save_dir=alpha_save_dir
            )

            all_results[alpha][sampling] = metrics

            print(f"\n  Results (alpha={alpha}, {sampling}):")
            print(f"  {metrics}")

    # save summary table
    _save_summary(all_results, alphas, samplings, save_dir, impl_name)

    return all_results


def _save_summary(all_results, alphas, samplings, save_dir, impl_name):
    """
    Save a clean summary JSON and print a formatted table.
    """
    # build serializable dict
    summary = {}
    for alpha in alphas:
        summary[str(alpha)] = {}
        for sampling in samplings:
            m = all_results[alpha][sampling]
            summary[str(alpha)][sampling] = {
                "accuracy":  round(m.accuracy, 2),
                "precision": round(m.precision, 2),
                "recall":    round(m.recall, 2),
                "f1":        round(m.f1, 2),
                "yes_ratio": round(m.yes_ratio, 2),
                "total":     m.total
            }

    summary_file = os.path.join(save_dir, impl_name, "summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Eval] Summary saved to {summary_file}")

    # print formatted table
    print("\n" + "="*80)
    print("POPE EVALUATION SUMMARY")
    print("="*80)
    header = f"{'Alpha':<8} {'Sampling':<14} {'Acc':>7} {'Prec':>7} "
    header += f"{'Rec':>7} {'F1':>7} {'Yes%':>7}"
    print(header)
    print("-"*80)

    for alpha in alphas:
        for sampling in samplings:
            m = all_results[alpha][sampling]
            print(
                f"{alpha:<8} {sampling:<14} "
                f"{m.accuracy:>7.2f} {m.precision:>7.2f} "
                f"{m.recall:>7.2f} {m.f1:>7.2f} "
                f"{m.yes_ratio:>7.2f}"
            )
        print("-"*80)