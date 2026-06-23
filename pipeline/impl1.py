import os
import torch
from PIL import Image
from typing import Dict, Optional, Tuple

from guidance.detector import ObjectDetector
from guidance.representation import RepresentationExtractor
from utils.cache_io import save_guidance, load_guidance, guidance_is_cached, save_guidance_logits, load_guidance_logits, guidance_logits_is_cached
from config import DEVICE, MAX_NEW_TOKENS, ALPHA
import torch.nn.functional as F


class Pipeline1:
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = ObjectDetector()
        self.extractor = RepresentationExtractor(model, tokenizer)

        # in-memory cache: avoids re-reading disk for same image in one run
        self._memory_cache: Dict[
            str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = {}

        self._logits_memory_cache: Dict[
            Tuple[str, str], torch.Tensor
        ] = {}

        self._cache_hits_memory   = 0
        self._cache_hits_disk     = 0
        self._cache_misses        = 0
        self._logits_hits_memory  = 0
        self._logits_hits_disk    = 0
        self._logits_misses       = 0

    def _get_image_key(
        self,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> str:
        if image_path is not None:
            return os.path.basename(image_path)
        thumb = image.resize((32, 32)).tobytes()
        return str(hash(thumb))

    def _compute_and_save_guidance(
        self,
        image: Image.Image,
        image_key: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run detection + SVD, save result to disk, return (V, V_neg).
        Called only when neither memory nor disk cache has this image.
        """
        detected, non_detected = self.detector.detect_with_negatives(image)

        V     = None
        V_neg = None

        if detected:
            rep_matrix = self.extractor.build_representation_matrix(detected)
            if rep_matrix is not None:
                self.model.lm_head.precompute(rep_matrix)
                V = self.model.lm_head.V.detach().cpu()

        
        if non_detected:
            neg_matrix = self.extractor.build_representation_matrix(non_detected)
            if neg_matrix is not None:
                self.model.lm_head.precompute_negative(neg_matrix)
                V_neg = self.model.lm_head.V_neg.detach().cpu()

        self.model.lm_head.reset()

        # persist to disk
        save_guidance(image_key, V, V_neg)

        return V, V_neg

    def _get_guidance(
        self,
        image: Image.Image,
        image_key: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Guidance lookup with three levels:
          1. memory cache  (fastest)
          2. disk cache    (fast, persistent across runs)
          3. compute       (slow, only when unseen image)
        """
        # level 1 — memory
        if image_key in self._memory_cache:
            self._cache_hits_memory += 1
            return self._memory_cache[image_key]

        # level 2 — disk
        if guidance_is_cached(image_key):
            result = load_guidance(image_key)
            if result is not None:
                self._cache_hits_disk += 1
                self._memory_cache[image_key] = result
                return result

        # level 3 — compute and save
        self._cache_misses += 1
        result = self._compute_and_save_guidance(image, image_key)
        self._memory_cache[image_key] = result
        return result

    def precompute_guidance_for_dataset(
        self,
        image_paths: list,
        image_dir: str
    ):
        """
        Precompute and persist guidance for all images.
        Skips images already cached on disk.
        """
        unique_paths = list(set(os.path.basename(p) for p in image_paths))

        # check how many are already on disk
        already_cached = sum(
            1 for p in unique_paths
            if guidance_is_cached(os.path.basename(p))
        )
        to_compute = len(unique_paths) - already_cached

        print(f"[Pipeline1] Guidance precompute: "
              f"{already_cached} already on disk, "
              f"{to_compute} to compute.")

        for i, fname in enumerate(unique_paths):
            key = os.path.basename(fname)

            # skip if already on disk
            if guidance_is_cached(key):
                continue

            image_path = os.path.join(image_dir, fname)
            try:
                image = Image.open(image_path).convert("RGB")
            except FileNotFoundError:
                print(f"[WARNING] Not found: {image_path}")
                save_guidance(key, None, None)
                continue

            self._compute_and_save_guidance(image, key)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(unique_paths)}] done...")

        print(f"[Pipeline1] Precompute complete.")

    def clear_memory_cache(self):
        """Clear in-memory cache only — disk cache is preserved."""
        self._memory_cache.clear()
        print("[Pipeline1] Memory cache cleared (disk cache preserved).")

    def cache_stats(self):
        total = (self._cache_hits_memory
                + self._cache_hits_disk
                + self._cache_misses)
        logits_total = (self._logits_hits_memory
                        + self._logits_hits_disk
                        + self._logits_misses)

        print(
            f"\n[Pipeline1] Guidance (V, V_neg) cache —\n"
            f"  memory hits: {self._cache_hits_memory}, "
            f"  disk hits:   {self._cache_hits_disk}, "
            f"  misses:      {self._cache_misses}, "
            f"  total:       {total}\n"
            f"[Pipeline1] Guidance logits cache —\n"
            f"  memory hits: {self._logits_hits_memory}, "
            f"  disk hits:   {self._logits_hits_disk}, "
            f"  misses:      {self._logits_misses}, "
            f"  total:       {logits_total}"
        )

    def _build_guidance_prompt(
        self,
        detected_objects: list,
        image: Image.Image,
        text: str
    ) -> dict:
        """
        Build the MARINE-style guidance input:
        injects detected objects as a text prefix before the question.
        """
        object_str = ", ".join(detected_objects)
        guidance_text = (
            f"The image contains only the following objects: {object_str}. Do not assume anything beyond these objects. Based solely on the list, {text}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": guidance_text},
                ],
            }
        ]

        formatted = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return self.processor(
            text=formatted,
            images=[image],
            return_tensors="pt"
        ).to(DEVICE, dtype=torch.bfloat16)


    @torch.no_grad()
    def _run_marine_forward(
        self,
        detected_objects: list,
        image: Image.Image,
        text: str,
        image_key: str
    ):
        """
        Run guided forward pass with detected objects injected as text prompt.
        Result cached in memory and on disk keyed by (image, prompt).
        """
        if not detected_objects or self.model.lm_head.mode != "combined":
            return

        cache_key = (image_key, text)

        # ── level 1: memory cache ───────────────────────────────────────────
        if cache_key in self._logits_memory_cache:
            self._logits_hits_memory += 1
            log_p_guided = self._logits_memory_cache[cache_key]
            self.model.lm_head.set_guidance_logits(
                log_p_guided.to(next(self.model.parameters()).device)
            )
            return

        # ── level 2: disk cache ─────────────────────────────────────────────
        if guidance_logits_is_cached(image_key, text):
            log_p_guided = load_guidance_logits(image_key, text)
            if log_p_guided is not None:
                self._logits_hits_disk += 1
                self._logits_memory_cache[cache_key] = log_p_guided
                self.model.lm_head.set_guidance_logits(
                    log_p_guided.to(next(self.model.parameters()).device)
                )
                return

        # ── level 3: compute ────────────────────────────────────────────────
        self._logits_misses += 1

        # disable modification during guided forward pass
        saved_mode = self.model.lm_head.mode
        self.model.lm_head.mode = "none"

        guidance_inputs = self._build_guidance_prompt(
            detected_objects, image, text
        )

        outputs = self.model(
            **guidance_inputs,
            return_dict=True
        )

        # take last token position logits
        last_logits  = outputs.logits[:, -1, :]              # B × vocab_size
        log_p_guided = F.log_softmax(last_logits.float(), dim=-1)

        # restore mode
        self.model.lm_head.mode = saved_mode

        # persist to disk and memory
        save_guidance_logits(image_key, text, log_p_guided.cpu())
        self._logits_memory_cache[cache_key] = log_p_guided.cpu()

        self.model.lm_head.set_guidance_logits(
            log_p_guided.to(next(self.model.parameters()).device)
        )

    def generate(
        self,
        text: str,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> str:

        # ── Step 1: get guidance ────────────────────────────────────────────
        image_key = self._get_image_key(image, image_path)
        V, V_neg = self._get_guidance(image, image_key)

        # ── Step 2: set V and V_neg on lm_head ─────────────────────────────
        self.model.lm_head.V     = V
        self.model.lm_head.V_neg = V_neg

        # ── Step 3: MARINE guided forward pass ─────────────────────────────
        if self.model.lm_head.mode == "combined":
            detected_key = f"{image_key}_detected"
            if detected_key in self._memory_cache:
                detected_objects = self._memory_cache[detected_key]
            else:
                detected_objects = self.detector.detect(image)
                self._memory_cache[detected_key] = detected_objects

            self._run_marine_forward(
                detected_objects=detected_objects,
                image=image,
                text=text,
                image_key=image_key      # ← pass image_key for cache keying
            )
        

        # ── Step 4: prepare original inputs ────────────────────────────────
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]

        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=formatted_text,
            images=[image],
            return_tensors="pt"
        ).to(DEVICE, dtype=torch.bfloat16)

        # ── Step 5: generate ────────────────────────────────────────────────
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

        # ── Step 6: reset ───────────────────────────────────────────────────
        self.model.lm_head.reset()

        return generated_text