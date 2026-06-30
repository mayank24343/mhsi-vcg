# models/marine_processor.py

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
from typing import Optional


class MarineLogitsProcessor(LogitsProcessor):
    """
    At each generation step:
      1. Run a forward pass with the guidance prompt + tokens generated so far
      2. Compute convex combination:
         log_p = gamma * log_p_guided + (1-gamma) * log_p_original
      3. Return combined logits for sampling

    This correctly conditions the guided distribution on all
    previously generated tokens, not just the initial prompt.
    """

    def __init__(
        self,
        model,
        guidance_input_ids: torch.Tensor,       # tokenized guidance prompt
        guidance_attention_mask: torch.Tensor,
        gamma: float = 0.7,
        alpha: float = 1.0,                     # SVD subspace strength
    ):
        self.model                   = model
        self.guidance_input_ids      = guidance_input_ids
        self.guidance_attention_mask = guidance_attention_mask
        self.gamma                   = gamma
        self.alpha                   = alpha

        # KV cache for guidance sequence — avoids reprocessing from scratch
        self.past_key_values = None
        self.is_first_step   = True

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,      # B × T — original tokens so far
        scores: torch.Tensor          # B × vocab_size — original logits
    ) -> torch.Tensor:

        # ── original log probs ──────────────────────────────────────────
        log_p_original = F.log_softmax(scores.float(), dim=-1)

        # ── guided log probs ────────────────────────────────────────────
        # disable SVD modification during this forward pass
        # so we get clean logits from the guidance prompt
        saved_mode = self.model.lm_head.mode
        self.model.lm_head.mode = "none"

        if self.is_first_step:
            # first step — run full guidance prompt forward pass
            # append any tokens generated so far (usually none at step 0)
            # build full guidance sequence:
            # [guidance_prompt_tokens] + [generated_tokens_so_far]

            # generated tokens so far — everything after the original prompt
            # at step 0 input_ids is just the original prompt, so no generated yet
            guidance_ids = self.guidance_input_ids.to(input_ids.device)
            guidance_mask = self.guidance_attention_mask.to(input_ids.device)

            outputs = self.model(
                input_ids=guidance_ids,
                attention_mask=guidance_mask,
                use_cache=True,
                return_dict=True
            )

            # store KV cache — will be reused for subsequent steps
            self.past_key_values = outputs.past_key_values
            self.is_first_step   = False

            # logits at last position
            last_logits = outputs.logits[:, -1, :]

        else:
            # subsequent steps — only pass the last generated token
            # reuse KV cache for efficiency
            last_token = input_ids[:, -1:]   # B × 1 — most recently sampled token

            outputs = self.model(
                input_ids=last_token,
                use_cache=True,
                past_key_values=self.past_key_values,
                return_dict=True
            )

            self.past_key_values = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

        # restore SVD mode
        self.model.lm_head.mode = saved_mode

        log_p_guided = F.log_softmax(last_logits.float(), dim=-1)

        # ── convex combination ──────────────────────────────────────────
        log_p_combined = (
            self.gamma       * log_p_guided
            + (1 - self.gamma) * log_p_original
        )

        """
        # ── SVD subspace correction on top ──────────────────────────────
        # applies the geometric boost/suppress from the weight matrix alignment
        W      = self.model.lm_head.original_lm_head.weight   # vocab_size × d
        W_norm = F.normalize(W.float(), dim=-1)

        if self.model.lm_head.V is not None:
            V = self.model.lm_head.V.to(W.device, dtype=W_norm.dtype)
            score_pos = ((W_norm @ V) ** 2).sum(dim=-1)          # vocab_size
            log_p_combined = (
                log_p_combined
                + self.alpha * score_pos.to(log_p_combined.dtype)
            )

        if self.model.lm_head.V_neg is not None:
            V_neg = self.model.lm_head.V_neg.to(W.device, dtype=W_norm.dtype)
            score_neg = ((W_norm @ V_neg) ** 2).sum(dim=-1)
            log_p_combined = (
                log_p_combined
                - self.alpha * score_neg.to(log_p_combined.dtype)
            )
        """

        out = F.log_softmax(log_p_combined, dim= -1)

        return out