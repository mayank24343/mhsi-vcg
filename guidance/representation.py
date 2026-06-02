import torch
from typing import List
from config import DEVICE, ALPHA, TOP_SVD_COMPONENTS, LVLM_MODEL_NAME


class RepresentationExtractor:
    """
    For each text string (object name or caption), 
    extract its dx1 representation from the LLM's
    final hidden layer before the LM head.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_text_representation(self, text: str) -> torch.Tensor:
        """
        Tokenize text, run through LLM, return final hidden state
        of the last token. Shape: d x 1

        Args:
            text: a single string e.g. "person" or "a dog sitting on a chair"
        """
        
        if ("Qwen2-VL" in LVLM_MODEL_NAME):
            
            # 2. Get the exact datatype of the embedding weights
            embed_dtype = self.model.language_model.embed_tokens.weight.dtype

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True
            ).to(next(self.model.language_model.parameters()).device)

        
            inputs_embeds = self.model.language_model.embed_tokens(inputs.input_ids).to(embed_dtype) 

            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask = inputs.attention_mask,
                output_hidden_states=True # need hidden states not just logits
            )

            # final layer hidden state: 1 x T x d
            # take last token: d
            last_idx = inputs.attention_mask[0].sum() - 1 # 1 1 1 0 then index is 2 sum is 3
            final_hidden = outputs.hidden_states[-1][0, last_idx].float()

        if ("SmolVLM" in LVLM_MODEL_NAME):
            # 2. Get the exact datatype of the embedding weights
            embed_dtype = self.model.model.text_model.embed_tokens.weight.dtype

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True
            ).to(next(self.model.model.text_model.parameters()).device)

        
            inputs_embeds = self.model.model.text_model.embed_tokens(inputs.input_ids).to(embed_dtype) 

            outputs = self.model.model.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask = inputs.attention_mask,
                output_hidden_states=True # need hidden states not just logits
            )

            # final layer hidden state: 1 x T x d
            # take last token: d
            last_idx = inputs.attention_mask[0].sum() - 1 # 1 1 1 0 then index is 2 sum is 3
            final_hidden = outputs.hidden_states[-1][0, last_idx].float()

        return final_hidden   # shape: d x 1

    @torch.no_grad()
    def build_representation_matrix(self, texts: List[str]) -> torch.Tensor:
        """
        Build M + k x d matrix

        Args:
            texts: list of object names or captions

        Returns:
            M + k x d tensor
        """
        if not texts:
            return None

        reps = []
        for text in texts:
            rep = self.get_text_representation(text)
            reps.append(rep)

        matrix = torch.stack(reps, dim=0)   # M + k × d
        print(f"[RepresentationExtractor] Matrix shape: {matrix.shape}")
        return matrix