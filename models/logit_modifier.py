import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SVDGuidedLogitModifier(nn.Module):
    def __init__(
        self,
        original_lm_head,
        alpha=1.0,
        mode="combined",
        top_k_pos=None,
        top_k_neg=None
    ):
        super().__init__()
        self.original_lm_head = original_lm_head
        self.alpha     = alpha
        self.mode      = mode      # "none", "token", "subspace"
        self.top_k_pos = top_k_pos
        self.top_k_neg = top_k_neg
        self.V         = None
        self.V_neg     = None
        self.pos_token_ids = None
        self.neg_token_ids = None

    def _compute_V(self, matrix, top_k):
        X = F.normalize(matrix.float(), dim=-1)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        if top_k is not None:
            Vh = Vh[:top_k, :]
        V = Vh.T
        V, _ = torch.linalg.qr(V)
        return V

    def precompute(self, matrix):
        self.V = self._compute_V(matrix, self.top_k_pos)

    def precompute_negative(self, matrix):
        self.V_neg = self._compute_V(matrix, self.top_k_neg)

    def precompute_tokens(self, detected, non_detected, tokenizer):
        self.pos_token_ids = self._get_token_ids(detected, tokenizer)
        self.neg_token_ids = self._get_token_ids(non_detected, tokenizer)

    def _get_token_ids(self, names, tokenizer):
        ids = set()
        for name in names:
            ids.update(tokenizer.encode(name, add_special_tokens=False))
        return list(ids)

    def reset(self):
        self.V             = None
        self.V_neg         = None
        self.pos_token_ids = None
        self.neg_token_ids = None

    def forward(self, hidden_states):
        logits = self.original_lm_head(hidden_states)

        """
        if self.mode == "none":
            return logits

        if self.mode == "token":
            if self.pos_token_ids:
                logits[:, :, self.pos_token_ids] += self.alpha
            if self.neg_token_ids:
                logits[:, :, self.neg_token_ids] -= self.alpha

        elif self.mode == "subspace":
            W      = self.original_lm_head.weight
            W_norm = F.normalize(W.float(), dim=-1)

            if self.V is not None:
                V = self.V.to(W.device, dtype=W_norm.dtype)
                score_pos = ((W_norm @ V) ** 2).sum(dim=-1)
                logits = logits + self.alpha * score_pos.to(logits.dtype)

            if self.V_neg is not None:
                V_neg = self.V_neg.to(W.device, dtype=W_norm.dtype)
                score_neg = ((W_norm @ V_neg) ** 2).sum(dim=-1)
                logits = logits - self.alpha * score_neg.to(logits.dtype)
        """

        return logits