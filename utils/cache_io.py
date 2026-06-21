import os
import json
import torch
import numpy as np
from typing import Optional, Tuple


GUIDANCE_CACHE_DIR = "cache/guidance"
POPE_CACHE_DIR     = "cache/pope_questions"


# ── Guidance cache ──────────────────────────────────────────────────────────

def guidance_cache_path(image_key: str) -> str:
    safe_key = image_key.replace("/", "_").replace("\\", "_")
    return os.path.join(GUIDANCE_CACHE_DIR, f"{safe_key}.json")


def save_guidance(
    image_key: str,
    V: Optional[torch.Tensor],
    V_neg: Optional[torch.Tensor]
):
    """
    Save V and V_neg tensors for an image to disk as JSON.
    Tensors stored as nested lists (float32).
    """
    os.makedirs(GUIDANCE_CACHE_DIR, exist_ok=True)
    path = guidance_cache_path(image_key)

    data = {
        "image_key": image_key,
        "V":     V.float().cpu().tolist() if V is not None else None,
        "V_neg": V_neg.float().cpu().tolist() if V_neg is not None else None,
    }

    with open(path, "w") as f:
        json.dump(data, f)


def load_guidance(
    image_key: str
) -> Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    Load V and V_neg for an image from disk.

    Returns:
        (V, V_neg) tuple if file exists, None if not cached yet.
    """
    path = guidance_cache_path(image_key)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    V     = torch.tensor(data["V"],     dtype=torch.float32) if data["V"]     is not None else None
    V_neg = torch.tensor(data["V_neg"], dtype=torch.float32) if data["V_neg"] is not None else None

    return V, V_neg


def guidance_is_cached(image_key: str) -> bool:
    return os.path.exists(guidance_cache_path(image_key))


# ── POPE question cache ──────────────────────────────────────────────────────

def pope_cache_path(sampling: str, num_images: int, questions_per_image: int) -> str:
    """
    Unique path per (sampling strategy, dataset size) combination.
    Ensures reproducibility — same file = same questions.
    """
    fname = f"pope_{sampling}_{num_images}img_{questions_per_image}q.json"
    return os.path.join(POPE_CACHE_DIR, fname)


def save_pope_questions(
    sampling: str,
    num_images: int,
    questions_per_image: int,
    questions: list
):
    """
    Save POPE questions to disk.
    questions: list of POPEQuestion dataclass instances → serialized as dicts.
    """
    os.makedirs(POPE_CACHE_DIR, exist_ok=True)
    path = pope_cache_path(sampling, num_images, questions_per_image)

    data = [
        {
            "image_id":    q.image_id,
            "image_file":  q.image_file,
            "object_name": q.object_name,
            "question":    q.question,
            "answer":      q.answer,
            "sampling":    q.sampling,
        }
        for q in questions
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[cache] POPE questions saved → {path}")


def load_pope_questions(
    sampling: str,
    num_images: int,
    questions_per_image: int
) -> Optional[list]:
    """
    Load POPE questions from disk if they exist.

    Returns:
        list of dicts if cached, None if not cached yet.
    """
    path = pope_cache_path(sampling, num_images, questions_per_image)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    print(f"[cache] POPE questions loaded ← {path} ({len(data)} questions)")
    return data


def pope_is_cached(
    sampling: str,
    num_images: int,
    questions_per_image: int
) -> bool:
    return os.path.exists(
        pope_cache_path(sampling, num_images, questions_per_image)
    )

def compute_metrics_from_pope_json(json_path: str) -> dict:
    """
    Compute POPE metrics directly from a saved detailed results JSON.

    Args:
        json_path: path to a detailed JSON file saved by run_pope_eval
                   e.g. "eval_results/impl1/alpha_0.0/alpha_0.0_random_detailed.json"

    Returns:
        dict with accuracy, precision, recall, f1, yes_ratio, total
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    tp = fp = tn = fn = yes_count = 0

    for entry in data:
        pred = entry["prediction"].strip().lower()
        gt   = entry["ground_truth"].strip().lower()

        if pred == "yes":
            yes_count += 1

        if   pred == "yes" and gt == "yes": tp += 1
        elif pred == "yes" and gt == "no":  fp += 1
        elif pred == "no"  and gt == "no":  tn += 1
        elif pred == "no"  and gt == "yes": fn += 1

    total = len(data)

    accuracy  = 100.0 * (tp + tn) / total          if total            > 0 else 0.0
    precision = 100.0 * tp / (tp + fp)              if (tp + fp)        > 0 else 0.0
    recall    = 100.0 * tp / (tp + fn)              if (tp + fn)        > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                                                     if (precision + recall) > 0 else 0.0
    yes_ratio = 100.0 * yes_count / total           if total            > 0 else 0.0

    metrics = {
        "accuracy":  round(accuracy,  2),
        "precision": round(precision, 2),
        "recall":    round(recall,    2),
        "f1":        round(f1,        2),
        "yes_ratio": round(yes_ratio, 2),
        "total":     total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    # pretty print
    print(f"\nPOPE Metrics — {os.path.basename(json_path)}")
    print(f"{'─'*40}")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall:    {recall:.2f}%")
    print(f"  F1:        {f1:.2f}%")
    print(f"  Yes Ratio: {yes_ratio:.2f}%")
    print(f"  Total:     {total}")
    print(f"  TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")
    print(f"{'─'*40}")

    return metrics

import hashlib

# ── Guidance logits cache ────────────────────────────────────────────────────

GUIDANCE_LOGITS_CACHE_DIR = "cache/guidance_logits"


def _make_logits_key(image_key: str, text: str) -> str:
    """
    Unique key for (image, prompt) pair.
    Uses image filename + md5 hash of prompt text.
    """
    prompt_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    safe_image_key = image_key.replace("/", "_").replace("\\", "_")
    return f"{safe_image_key}__{prompt_hash}"


def guidance_logits_cache_path(image_key: str, text: str) -> str:
    key = _make_logits_key(image_key, text)
    return os.path.join(GUIDANCE_LOGITS_CACHE_DIR, f"{key}.json")


def save_guidance_logits(
    image_key: str,
    text: str,
    log_p_guided: torch.Tensor
):
    """
    Save guidance logits for a (image, prompt) pair to disk.

    Args:
        image_key:    image filename
        text:         the guidance prompt text
        log_p_guided: B × vocab_size tensor (log softmax)
    """
    os.makedirs(GUIDANCE_LOGITS_CACHE_DIR, exist_ok=True)
    path = guidance_logits_cache_path(image_key, text)

    data = {
        "image_key": image_key,
        "text":      text,
        "logits":    log_p_guided.float().cpu().tolist()
    }

    with open(path, "w") as f:
        json.dump(data, f)


def load_guidance_logits(
    image_key: str,
    text: str
) -> Optional[torch.Tensor]:
    """
    Load guidance logits for a (image, prompt) pair from disk.

    Returns:
        B × vocab_size tensor or None if not cached
    """
    path = guidance_logits_cache_path(image_key, text)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    return torch.tensor(data["logits"], dtype=torch.float32)


def guidance_logits_is_cached(image_key: str, text: str) -> bool:
    return os.path.exists(guidance_logits_cache_path(image_key, text))