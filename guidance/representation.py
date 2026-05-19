import torch
import torch.nn.functional as F
from typing import List


class RepresentationExtractor:
    """
    For each text string (object name or caption),
    extract its d×1 representation from the LLM's
    final hidden layer before the LM head.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_text_representation(self, text: str) -> torch.Tensor:
        """
        Tokenize text, run through LLM, return final hidden state
        of the last token. Shape: d

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
            output_hidden_states=True
        )

        # final layer hidden state: 1 × T × d
        # take last token: d
        final_hidden = outputs.hidden_states[-1][0, -1, :]

        return final_hidden   # shape: d

    @torch.no_grad()
    def build_representation_matrix(self, texts: List[str]) -> torch.Tensor:
        """
        Build M × d matrix from list of text strings.

        Args:
            texts: list of object names or captions

        Returns:
            M × d tensor
        """
        if not texts:
            return None

        reps = []
        for text in texts:
            rep = self.get_text_representation(text)
            reps.append(rep)

        matrix = torch.stack(reps, dim=0)   # M × d
        print(f"[RepresentationExtractor] Matrix shape: {matrix.shape}")
        return matrix