import torch
from typing import List


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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(next(self.model.language_model.parameters()).device)

        outputs = self.model.language_model(
            input_ids=inputs.input_ids,
            output_hidden_states=True # need hidden states not just logits
        )

        # final layer hidden state: 1 x T x d
        # take last token: d
        last_idx = inputs.attention_mask[0].sum() - 1 # 1 1 1 0 then index is 2 sum is 3
        final_hidden = outputs.hidden_states[-1][0, last_idx] 

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