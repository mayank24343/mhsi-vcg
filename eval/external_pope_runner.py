import os
import json
import torch
import copy
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional

from eval.pope_metrics import compute_metrics, POPEMetrics
from eval.pope_runner import extract_yes_no
from config import DEVICE


def load_jsonl_or_json(path: str) -> list:
    """
    Load a file that is either:
      - JSONL: one JSON object per line  (pope question files)
      - JSON:  a single JSON array       (detection files)
    """
    with open(path, "r") as f:
        content = f.read().strip()

    # try JSON array first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # fall back to JSONL
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_external_questions(question_file: str) -> List[dict]:
    """
    Load POPE questions from an external JSONL file.

    Each entry:
        {"question_id": 1, "image": "...", "text": "...", "label": "yes"}

    Returns list of dicts with standardized keys.
    """
    raw = load_jsonl_or_json(question_file)

    questions = []
    for entry in raw:
        questions.append({
            "question_id": entry["question_id"],
            "image_file":  entry["image"],
            "text":        entry["text"],
            "answer":      entry["label"].strip().lower()
        })

    print(f"[external] Loaded {len(questions)} questions from {os.path.basename(question_file)}")
    return questions


def load_external_detections(
    detr_file: str,
    ram_file: str
) -> Dict[str, list]:
    """
    Load DETR and RAM++ detections and compute their intersection per image.

    Returns:
        dict: image_filename → list of detected object names (intersection)
    """
    detr_raw = load_jsonl_or_json(detr_file)
    ram_raw  = load_jsonl_or_json(ram_file)

    # build lookup: image → set of objects
    detr_map: Dict[str, set] = {}
    for entry in detr_raw:
        fname = entry["image"]
        detr_map[fname] = set(entry["objects"])

    ram_map: Dict[str, set] = {}
    for entry in ram_raw:
        fname = entry["image"]
        ram_map[fname] = set(entry["objects"])

    # intersection per image
    all_images = set(detr_map.keys()) | set(ram_map.keys())
    detection_map: Dict[str, list] = {}

    for fname in all_images:
        detr_objs = detr_map.get(fname, set())
        ram_objs  = ram_map.get(fname,  set())

        intersection = detr_objs & ram_objs

        # fallback: if intersection empty use DETR alone
        if not intersection:
            intersection = detr_objs

        detection_map[fname] = list(intersection)

    print(f"[external] Loaded detections for {len(detection_map)} images")
    return detection_map


def build_vvt_from_detections(
    pipeline,
    detection_map: Dict[str, list],
    image_dir: str
):
    """
    For each image, compute V and V_neg from external detections
    and save to guidance cache.

    Skips images already in cache.

    Args:
        pipeline:      Pipeline1 instance
        detection_map: image_filename → detected objects list
        image_dir:     path to COCO val images
    """
    from utils.cache_io import save_guidance, guidance_is_cached
    from guidance.detector import COCO_VALID_CLASSES

    unique_images = list(detection_map.keys())
    already_cached = sum(1 for f in unique_images if guidance_is_cached(f))
    to_compute = len(unique_images) - already_cached

    print(f"[external] SVD precompute: "
          f"{already_cached} already cached, {to_compute} to compute.")

    for fname in tqdm(unique_images, desc="[external] Computing SVD guidance"):
        if guidance_is_cached(fname):
            continue

        detected = detection_map.get(fname, [])
        detected_set = set(detected)

        non_detected = [
            c for c in COCO_VALID_CLASSES
            if c not in detected_set
        ]

        V     = None
        V_neg = None

        if detected:
            rep = pipeline.extractor.build_representation_matrix(detected)
            if rep is not None:
                pipeline.model.lm_head.precompute(rep)
                V = pipeline.model.lm_head.V.detach().cpu()

        if non_detected:
            rep_neg = pipeline.extractor.build_representation_matrix(non_detected)
            if rep_neg is not None:
                pipeline.model.lm_head.precompute_negative(rep_neg)
                V_neg = pipeline.model.lm_head.V_neg.detach().cpu()

        pipeline.model.lm_head.reset()
        save_guidance(fname, V, V_neg)

    print("[external] SVD precompute complete.")


def run_external_pope_eval(
    pipeline,
    questions: List[dict],
    image_dir: str,
    detection_map: Dict[str, list],
    alpha: float,
    sampling_name: str,
    save_dir: str
) -> POPEMetrics:
    """
    Run POPE evaluation using externally loaded questions and detections.

    Args:
        pipeline:      Pipeline1 instance
        questions:     list of question dicts from load_external_questions
        image_dir:     path to COCO val images
        detection_map: image_filename → detected objects
        alpha:         current alpha value
        sampling_name: "random", "popular", or "adversarial"
        save_dir:      where to save detailed results
    """
    # store detected objects in pipeline memory cache
    # so _run_marine_forward can access them without re-detecting
    for fname, objs in detection_map.items():
        pipeline._memory_cache[f"{fname}_detected"] = objs

    predictions    = []
    ground_truths  = []
    detailed       = []

    for q in tqdm(questions, desc=f"  [{sampling_name}] alpha={alpha}"):
        image_path = os.path.join(image_dir, q["image_file"])

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARNING] Not found: {image_path}")
            continue

        raw_answer = pipeline.generate(
            text=q["text"],
            image=image,
            image_path=q["image_file"]
        )

        pred = extract_yes_no(raw_answer)
        predictions.append(pred)
        ground_truths.append(q["answer"])

        detailed.append({
            "question_id": q["question_id"],
            "image_file":  q["image_file"],
            "question":    q["text"],
            "ground_truth":q["answer"],
            "raw_answer":  raw_answer,
            "prediction":  pred,
            "correct":     pred == q["answer"]
        })

    metrics = compute_metrics(predictions, ground_truths)

    # save detailed results
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(
        save_dir,
        f"alpha_{alpha}_{sampling_name}_detailed.json"
    )
    with open(out_file, "w") as f:
        json.dump(detailed, f, indent=2)

    print(f"  [Saved] {out_file}")
    return metrics


def run_full_external_evaluation(
    model,
    tokenizer,
    processor,
    pipeline_class,
    question_files: Dict[str, str],    # {"random": path, "popular": path, "adversarial": path}
    detr_file: str,
    ram_file: str,
    image_dir: str,
    alphas: List[float],
    save_dir: str = "eval_results",
    impl_name: str = "impl1"
):
    """
    Full evaluation using external question and detection files.

    Args:
        model, tokenizer, processor: loaded model
        pipeline_class: Pipeline1 or Pipeline2
        question_files: dict mapping sampling name → path to question JSONL
        detr_file:      path to coco_detr.json
        ram_file:       path to coco_ram.json
        image_dir:      path to COCO val images
        alphas:         list of alpha values to sweep
        save_dir:       root output directory
        impl_name:      label for output files
    """
    # ── load external data ──────────────────────────────────────────────
    print("\n[external] Loading questions and detections...")

    question_sets = {
        sampling: load_external_questions(path)
        for sampling, path in question_files.items()
    }

    detection_map = load_external_detections(detr_file, ram_file)

    # ── precompute SVD guidance once ────────────────────────────────────
    print("\n[external] Precomputing SVD guidance from external detections...")
    init_pipeline = pipeline_class(model, tokenizer, processor)
    #build_vvt_from_detections(init_pipeline, detection_map, image_dir)
    shared_cache = init_pipeline._memory_cache

    # ── sweep alphas ────────────────────────────────────────────────────
    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Evaluating alpha = {alpha}")
        print(f"{'='*60}")

        model.lm_head.alpha = alpha

        pipeline = pipeline_class(model, tokenizer, processor)
        pipeline._memory_cache = shared_cache

        all_results[alpha] = {}

        for sampling, questions in question_sets.items():
            print(f"\n  Sampling: {sampling}")

            alpha_save_dir = os.path.join(
                save_dir, impl_name, f"alpha_{alpha}"
            )

            metrics = run_external_pope_eval(
                pipeline=pipeline,
                questions=questions,
                image_dir=image_dir,
                detection_map=detection_map,
                alpha=alpha,
                sampling_name=sampling,
                save_dir=alpha_save_dir
            )

            all_results[alpha][sampling] = metrics
            print(f"\n  {metrics}")

    _save_summary(all_results, alphas, list(question_sets.keys()), save_dir, impl_name)
    return all_results


def _save_summary(all_results, alphas, samplings, save_dir, impl_name):
    from eval.pope_runner import _save_summary as base_save_summary
    base_save_summary(all_results, alphas, samplings, save_dir, impl_name)