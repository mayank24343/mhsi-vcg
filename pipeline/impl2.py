import torch
from PIL import Image
from typing import List

from guidance.detector import ObjectDetector
from guidance.captioner import ImageCaptioner
from guidance.representation import RepresentationExtractor
from config import DEVICE, MAX_NEW_TOKENS


class Pipeline2:
    """
    Implementation 2:
    (Objects from DETR ∩ RAM++) + (Captions from BLIP)
    → combined representations → SVD → modified LM head → generate
    """

    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = ObjectDetector()
        self.captioner = ImageCaptioner(use_blip2=False)
        self.extractor = RepresentationExtractor(model, tokenizer)

    def generate(self, text: str, image: Image.Image) -> str:
        """
        Args:
            text: question or prompt string
            image: PIL image

        Returns:
            generated answer string
        """
        print("\n[Pipeline2] Starting generation...")

        # ── Step 1: detect objects ──────────────────────────────────────
        detected_objects = self.detector.detect(image)

        # ── Step 2: generate captions ───────────────────────────────────
        captions = self.captioner.caption(image, num_captions=3)

        # ── Step 3: combine all texts ───────────────────────────────────
        all_texts = detected_objects + captions
        # e.g. ["person", "chair", "laptop", "a man sitting at a desk", ...]

        print(f"[Pipeline2] Total texts for representation: {len(all_texts)}")
        print(f"  Objects ({len(detected_objects)}): {detected_objects}")
        print(f"  Captions ({len(captions)}): {captions}")

        # ── Step 4: build (M+Q) × d representation matrix ──────────────
        if all_texts:
            rep_matrix = self.extractor.build_representation_matrix(all_texts)
            # (M+Q) × d
        else:
            rep_matrix = None

        # ── Step 5: precompute SVD ──────────────────────────────────────
        if rep_matrix is not None:
            self.model.language_model.lm_head.precompute(rep_matrix)

        # ── Step 6: prepare inputs ──────────────────────────────────────
        prompt = f"USER: <image>\n{text} ASSISTANT:"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        # ── Step 7: generate ────────────────────────────────────────────
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        answer = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        # ── Step 8: reset ───────────────────────────────────────────────
        self.model.language_model.lm_head.reset()

        print(f"[Pipeline2] Answer: {answer}")
        return answer