import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
from config import DEVICE


class ImageCaptioner:
    """
    Generates captions for an image using BLIP / BLIP-2.
    Used in Implementation 2 to augment the representation matrix.
    """

    def __init__(self, use_blip2=False):
        if use_blip2:
            print("[Captioner] Loading BLIP-2...")
            self.processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16
            ).to(DEVICE)
        else:
            print("[Captioner] Loading BLIP...")
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(DEVICE)

        self.model.eval()
        print("[Captioner] Loaded.")

    @torch.no_grad()
    def caption(self, image: Image.Image, num_captions: int = 3) -> list:
        """
        Generate multiple diverse captions for the image.

        Args:
            image: PIL image
            num_captions: how many captions to generate

        Returns:
            list of caption strings
        """
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        # generate multiple captions with sampling for diversity
        outputs = self.model.generate(
            **inputs,
            num_return_sequences=num_captions,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=50
        )

        captions = [
            self.processor.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

        print(f"[Captioner] Captions: {captions}")
        return captions