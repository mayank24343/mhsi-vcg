import torch
from PIL import Image
from typing import List

from guidance.detector import ObjectDetector
from guidance.representation import RepresentationExtractor
from config import DEVICE, MAX_NEW_TOKENS, ALPHA


class Pipeline1:
    """
    Implementation 1:
    Objects from DETR ∩ RAM++ → representations → SVD → modified LM head → generate
    """

    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = ObjectDetector()
        self.extractor = RepresentationExtractor(model, tokenizer)

    def generate(self, text: str, image: Image.Image) -> str:
        """
        Args:
            text: question or prompt string
            image: PIL image

        Returns:
            generated answer string
        """
        print("\n[Pipeline1] Starting generation...")

        # ── Step 1: detect objects ──────────────────────────────────────
        detected_objects = self.detector.detect(image)

        # ── Step 2: get LLM representations ────────────────────────────
        if detected_objects:
            rep_matrix = self.extractor.build_representation_matrix(
                detected_objects
            )
            # M × d
        else:
            print("[Pipeline1] No objects detected, skipping SVD guidance.")
            rep_matrix = None

        # ── Step 3: precompute SVD and set VVᵀ in LM head ──────────────
        if rep_matrix is not None:
            self.model.lm_head.precompute(rep_matrix)

        # ── Step 4: prepare inputs for LLaVA ───────────────────────────
        # LLaVA expects a specific prompt format
        # ── Step 4: prepare inputs for Qwen2-VL ───────────────────────
        print (ALPHA)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
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

        # ── Step 5: generate ────────────────────────────────────────────
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

        answer = generated_text

        # ── Step 6: reset LM head for next image ────────────────────────
        self.model.lm_head.reset()

        print(f"[Pipeline1] Answer: {answer}")
        return answer