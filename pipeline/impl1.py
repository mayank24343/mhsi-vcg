import os
import torch
from PIL import Image
from typing import Dict, Optional, Tuple

from guidance.detector import ObjectDetector
from guidance.representation import RepresentationExtractor
from config import DEVICE, MAX_NEW_TOKENS, ALPHA


class Pipeline1:
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = ObjectDetector()
        self.extractor = RepresentationExtractor(model, tokenizer)

        # cache: image_key → (V, V_neg) both on CPU, either can be None
        self._guidance_cache: Dict[
            str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = {}

        self._cache_hits = 0
        self._cache_misses = 0

    def _get_image_key(
        self,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> str:
        if image_path is not None:
            return os.path.basename(image_path)
        thumb = image.resize((32, 32)).tobytes()
        return str(hash(thumb))

    def _compute_guidance(
        self,
        image: Image.Image
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run detection, build V and V_neg, return (V, V_neg) on CPU.
        Called only on cache miss.
        """
        # get detected and non-detected objects
        detected, non_detected = self.detector.detect_with_negatives(image)

        V = None
        V_neg = None

        # ── positive subspace (detected objects) ───────────────────────
        if detected:
            rep_matrix = self.extractor.build_representation_matrix(detected)
            if rep_matrix is not None:
                self.model.lm_head.precompute(rep_matrix)
                V = self.model.lm_head.V.detach().cpu()

        # ── negative subspace (non-detected COCO objects) ───────────────
        if non_detected:
            neg_matrix = self.extractor.build_representation_matrix(
                non_detected
            )
            if neg_matrix is not None:
                self.model.lm_head.precompute_negative(neg_matrix)
                V_neg = self.model.lm_head.V_neg.detach().cpu()

        # reset lm_head — pipeline sets V/V_neg explicitly per generate call
        self.model.lm_head.reset()

        return V, V_neg

    def precompute_guidance_for_dataset(
        self,
        image_paths: list,
        image_dir: str
    ):
        unique_paths = list(set(os.path.basename(p) for p in image_paths))
        print(f"[Pipeline1] Precomputing guidance for "
              f"{len(unique_paths)} unique images...")

        for i, fname in enumerate(unique_paths):
            key = os.path.basename(fname)
            if key in self._guidance_cache:
                continue

            image_path = os.path.join(image_dir, fname)
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"[WARNING] Not found: {image_path}")
                self._guidance_cache[key] = (None, None)
                continue

            V, V_neg = self._compute_guidance(image)
            self._guidance_cache[key] = (V, V_neg)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(unique_paths)}] cached...")

        print(f"[Pipeline1] Done. {len(self._guidance_cache)} images cached.")

    def clear_cache(self):
        self._guidance_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        print("[Pipeline1] Cache cleared.")

    def cache_stats(self):
        total = self._cache_hits + self._cache_misses
        hit_rate = 100 * self._cache_hits / total if total > 0 else 0
        print(
            f"[Pipeline1] Cache — "
            f"hits: {self._cache_hits}, "
            f"misses: {self._cache_misses}, "
            f"hit rate: {hit_rate:.1f}%"
        )

    def generate(
        self,
        text: str,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> str:
        # ── Step 1: look up or compute guidance ────────────────────────
        image_key = self._get_image_key(image, image_path)

        if image_key in self._guidance_cache:
            V, V_neg = self._guidance_cache[image_key]
            self._cache_hits += 1
        else:
            self._cache_misses += 1
            V, V_neg = self._compute_guidance(image)
            self._guidance_cache[image_key] = (V, V_neg)

        # ── Step 2: set V and V_neg on lm_head ─────────────────────────
        self.model.lm_head.V     = V      # None is fine — lm_head checks
        self.model.lm_head.V_neg = V_neg

        # ── Step 3: prepare inputs ──────────────────────────────────────
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": text},
                ],
            }
        ]

        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[formatted_text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        # ── Step 4: generate ────────────────────────────────────────────
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0
            )

        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        # ── Step 5: reset lm_head ───────────────────────────────────────
        self.model.lm_head.reset()
        print(f"[Pipeline1] Answer: {generated_text}")
        return generated_text