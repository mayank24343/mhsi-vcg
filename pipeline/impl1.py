import os
import torch
from PIL import Image
from typing import Dict, Optional

from guidance.detector import ObjectDetector
from guidance.representation import RepresentationExtractor
from config import DEVICE, MAX_NEW_TOKENS, ALPHA


class Pipeline1:
    """
    Implementation 1:
    Objects from DETR -> representations -> SVD -> modified LM head -> generate

    Guidance (detection + SVD) is computed ONCE per unique image
    and cached in memory for all subsequent queries on the same image.
    Cache stores V (d x k) the orthonormal basis of the object subspace.
    """

    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = ObjectDetector()
        self.extractor = RepresentationExtractor(model, tokenizer)

        # cache: image_key ->  V tensor (d × k) or None
        self._guidance_cache: Dict[str, Optional[torch.Tensor]] = {}

        self._cache_hits = 0
        self._cache_misses = 0

    def _get_image_key(
        self,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> str:
        """
        Unique key for this image.
        Prefer filename if provided — fast and reliable.
        Falls back to pixel hash if no path given.
        """
        if image_path is not None:
            return os.path.basename(image_path)

        thumb = image.resize((32, 32)).tobytes()
        return str(hash(thumb))

    def _compute_V(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        Run detection + representation extraction + SVD.
        Returns V (d x k) or None if no objects detected.
        This is the expensive part — called only on cache miss.
        """
        # Step 1: detect objects
        detected_objects = self.detector.detect(image)

        if not detected_objects:
            print("[Pipeline1] No objects detected, skipping SVD guidance.")
            return None

        # Step 2: get LLM representations → M x d
        rep_matrix = self.extractor.build_representation_matrix(
            detected_objects
        )

        if rep_matrix is None:
            return None

        # Step 3: compute SVD via lm_head, then read V back
        # precompute() runs SVD and stores V on self.model.lm_head
        self.model.lm_head.precompute(rep_matrix)
        V = self.model.lm_head.V   # d x k

        # detach and move to CPU for storage
        # (will be moved back to correct device in lm_head.forward)
        return V.detach().cpu()

    def precompute_guidance_for_dataset(
        self,
        image_paths: list,
        image_dir: str
    ):
        """
        Precompute and cache V for all images before evaluation starts.
        Call this ONCE before the eval loop.

        Args:
            image_paths: list of image filenames
            image_dir:   root directory containing the images
        """
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
                self._guidance_cache[key] = None
                continue

            V = self._compute_V(image)
            self._guidance_cache[key] = V

            # reset lm_head between images
            self.model.lm_head.reset()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(unique_paths)}] cached...")

        print(f"[Pipeline1] Done. {len(self._guidance_cache)} images cached.")

    def clear_cache(self):
        """Free memory — call between experiments or alpha sweeps."""
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
            f"hit rate: {hit_rate:.1f}%, "
            f"size: {len(self._guidance_cache)} images"
        )

    def generate(
        self,
        text: str,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> str:
        """
        Args:
            text:       question or prompt string
            image:      PIL image
            image_path: filename used as cache key — pass whenever available

        Returns:
            generated answer string
        """
        
        # cache because the test take too long to run (7-8 vs 1-2 seconds)

        image_key = self._get_image_key(image, image_path)

        if image_key in self._guidance_cache:
            V = self._guidance_cache[image_key]
            self._cache_hits += 1
        else:
            self._cache_misses += 1
            V = self._compute_V(image)
            self._guidance_cache[image_key] = V
        
        self.model.lm_head.V = V   

        # input template for qwen
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

        # generate output
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

        # reset before next iteration
        self.model.lm_head.reset()

        print(f"[Pipeline1] Answer: {generated_text}")
        return generated_text