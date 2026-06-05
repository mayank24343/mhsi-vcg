import torch

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    AutoModelForImageTextToText,
    SmolVLMProcessor, 
    SmolVLMForConditionalGeneration
)

from models.lm_head import SVDGuidedLMHead
from config import DEVICE, ALPHA, TOP_SVD_COMPONENTS
from models.logit_modifier import SVDGuidedLogitModifier


def load_model(model_name: str):

    print(f"[loader] Loading {model_name}...")
    if ("Qwen2-VL" in model_name):    
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(DEVICE)

        processor = AutoProcessor.from_pretrained(model_name)

        tokenizer = processor.tokenizer
    
        original_lm_head = model.lm_head
        model.lm_head = SVDGuidedLogitModifier(
            original_lm_head,
            alpha=ALPHA,
            top_k_pos=TOP_SVD_COMPONENTS,
            top_k_neg=TOP_SVD_COMPONENTS
        )

        """
        SVDGuidedLMHead(
            original_lm_head,
            alpha=ALPHA,
            top_k=TOP_SVD_COMPONENTS
        )
        """

        model.eval()
        print("[loader] Model loaded and LM head replaced.")

        return model, tokenizer, processor

    if ("SmolVLM" in model_name):
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(DEVICE)

        processor = SmolVLMProcessor.from_pretrained(model_name)
        tokenizer = processor.tokenizer

        print(f"[loader] LM head: {type(model.lm_head)}")

        original_lm_head = model.lm_head
        model.lm_head = SVDGuidedLogitModifier(
            original_lm_head,
            alpha=ALPHA,
            top_k_pos=TOP_SVD_COMPONENTS,
            top_k_neg=TOP_SVD_COMPONENTS
        )

        """
        SVDGuidedLMHead(
            original_lm_head,
            alpha=ALPHA,
            top_k=TOP_SVD_COMPONENTS
        )
        """

        return model, tokenizer, processor


    return model, tokenizer, processor
"""

import torch
from transformers import SmolVLMProcessor, SmolVLMForConditionalGeneration
from models.lm_head import SVDGuidedLMHead
from config import DEVICE, ALPHA, TOP_SVD_COMPONENTS


def load_model(model_name: str):
    print(f"[loader] Loading {model_name}...")

    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    processor = SmolVLMProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    print(f"[loader] LM head: {type(model.lm_head)}")

    original_lm_head = model.lm_head
    model.lm_head = SVDGuidedLMHead(
        original_lm_head,
        alpha=ALPHA,
        top_k=TOP_SVD_COMPONENTS
    )

    model.eval()
    print("[loader] Model loaded and LM head replaced.")
    return model, tokenizer, processor

"""