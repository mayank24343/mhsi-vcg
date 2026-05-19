import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from models.lm_head import SVDGuidedLMHead
from config import DEVICE, ALPHA, TOP_SVD_COMPONENTS


def load_model(model_name: str):
    """
    Load LLaVA and wrap its LM head with SVDGuidedLMHead.

    Returns:
        model, tokenizer, processor
    """
    print(f"[loader] Loading {model_name}...")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    #print(model)

    # see all named modules and their paths
    
    # output: language_model.lm_head <class 'torch.nn.Linear'>

    # replace lm_head with our modified version
    original_lm_head = model.lm_head
    model.lm_head = SVDGuidedLMHead(
        original_lm_head,
        alpha=ALPHA,
        top_k=TOP_SVD_COMPONENTS
    )

    model.eval()
    print("[loader] Model loaded and LM head replaced.")
    return model, tokenizer, processor