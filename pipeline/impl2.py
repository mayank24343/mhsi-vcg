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
    (Objects from DETR) + (Captions from BLIP)
    -> combined representations -> SVD -> modified LM head -> generate
    """

    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = ObjectDetector()
        self.captioner = ImageCaptioner()
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

        # detection, m objects
        detected_objects = self.detector.detect(image)

        # captioning, k captions
        captions = self.captioner.caption(image)

        # m + k total
        all_texts = detected_objects + captions

        print(f"[Pipeline2] Total texts for representation: {len(all_texts)}")
        print(f"  Objects ({len(detected_objects)}): {detected_objects}")
        print(f"  Captions ({len(captions)}): {captions}")

        # M + k x d matrix
        if all_texts:
            rep_matrix = self.extractor.build_representation_matrix(all_texts)
        else:
            rep_matrix = None

        # set matrix for this iteration
        if rep_matrix is not None:
            self.model.lm_head.precompute(rep_matrix)

        # input template for qwen
        #print (ALPHA)
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

        answer = generated_text

        # reset for next image
        self.model.lm_head.reset()

        print(f"[Pipeline2] Answer: {answer}")
        return answer