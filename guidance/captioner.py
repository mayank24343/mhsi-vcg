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
    Generates captions for an image using BLIP. 
    Used in Implementation 2 to augment the representation matrix.
    """

    def __init__(self):
    
        print("[Captioner] Loading BLIP...")
        self.processor2 = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model2 = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE)

        self.model2.eval()
        print("[Captioner] Loaded.")

        # blip 2 used to live here as model1 processor1, sadly it was too big for this device to run

    @torch.no_grad()
    def caption(self, image: Image.Image) -> list:
        """
        Generate multiple captions for the image but currently only one.

        Args:
            image: PIL image

        Returns:
            list of caption strings
        """

        inputs2 = self.processor2(
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        # generate multiple captions with sampling for diversity
        outputs2 = self.model2.generate(
            **inputs2,
            do_sample=False, #greedy so reproducible
            max_new_tokens=50 #caption max length
        )

        
        captions = [ self.processor2.decode(o, skip_special_tokens=True) for o in outputs2]

        print(f"[Captioner] Captions: {captions}")
        return captions