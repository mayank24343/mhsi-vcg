import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SVDGuidedLogitModifier(nn.Module):
    """
    Three modes:
      "token"    — directly boost/suppress object token logits
      "subspace" — boost/suppress via LM head weight alignment with SVD subspace
      "combined" — MARINE convex combination + SVD subspace correction
    """

    def __init__(
        self,
        original_lm_head: nn.Linear,
        alpha: float = 1.0,
        gamma: float = 0.7,           # MARINE guidance strength (convex weight)
        mode: str = "combined",
        top_k_pos: int = 3,
        top_k_neg: int = 3,
    ):
        super().__init__()
        self.original_lm_head = original_lm_head
        self.alpha    = alpha     # SVD subspace amplification strength
        self.gamma    = gamma     # MARINE convex combination weight
        self.mode     = mode
        self.top_k_pos = top_k_pos
        self.top_k_neg = top_k_neg

        # SVD subspace — set by precompute()
        self.V     = None   # d × k   positive subspace
        self.V_neg = None   # d × k'  negative subspace

        # MARINE guidance — set by set_guidance_logits()
        # stored as log_softmax of guided forward pass output
        self.guidance_logits = None   # B × vocab_size

        # token mode
        self.pos_token_ids = None
        self.neg_token_ids = None

    # ── SVD precomputation (unchanged) ──────────────────────────────────

    def _compute_V(
        self,
        representation_matrix: torch.Tensor,
        top_k: Optional[int]
    ) -> torch.Tensor:
        X = F.normalize(representation_matrix.float(), dim=-1)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        if top_k is not None:
            Vh = Vh[:top_k, :]
        V = Vh.T
        V, _ = torch.linalg.qr(V)
        return V

    def precompute(self, representation_matrix: torch.Tensor):
        self.V = self._compute_V(representation_matrix, self.top_k_pos)

    def precompute_negative(self, representation_matrix: torch.Tensor):
        self.V_neg = self._compute_V(representation_matrix, self.top_k_neg)

    def precompute_tokens(
        self,
        detected_objects: list,
        non_detected_objects: list,
        tokenizer
    ):
        self.pos_token_ids = self._get_token_ids(detected_objects, tokenizer)
        self.neg_token_ids = self._get_token_ids(non_detected_objects, tokenizer)

    def _get_token_ids(self, object_names: list, tokenizer) -> list:
        token_ids = set()
        for name in object_names:
            ids = tokenizer.encode(name, add_special_tokens=False)
            token_ids.update(ids)
        return list(token_ids)

    # ── MARINE guidance ──────────────────────────────────────────────────

    def set_guidance_logits(self, guidance_logits: torch.Tensor):
        """
        Store the log-softmax logits from the guided forward pass.
        Called by the pipeline before generation.

        Args:
            guidance_logits: B × vocab_size  (log softmax)
        """
        self.guidance_logits = guidance_logits

    def reset(self):
        self.V               = None
        self.V_neg           = None
        self.guidance_logits = None
        self.pos_token_ids   = None
        self.neg_token_ids   = None

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: B × T × d
        Returns:
            logits: B × T × vocab_size
        """

        """
        if self.V is not None:

            # match device/dtype dynamically
            V = self.V.to(
                hidden_states.device,
                dtype=hidden_states.dtype
            )

            # hidden_states: B x T x d
            # V: d x k
            #print("before modification")
            #print(hidden_states.shape)
            h_original = hidden_states

            coeffs = h_original @ V          # B x T x k

            amplification = coeffs @ V.T        # B x T x d

            hidden_states = (
                self.alpha*h_original
                + (1-self.alpha)* amplification
            )
           #print(hidden_states.shape, V.shape)
            #print("after modification")
            #print(hidden_states.shape)
            
            if self.V_neg is not None:
                V_neg = self.V_neg.to(
                    hidden_states.device,
                    dtype=hidden_states.dtype
                )
                neg_projection = (h_original @ V_neg) @ V_neg.T  # B × T × d
                hidden_states = hidden_states - (1 - self.alpha) * neg_projection
    
        logits = self.original_lm_head(hidden_states)
        """

        """
        if self.mode == "token":
            logits = self._token_mode(logits)

        elif self.mode == "subspace":
            logits = self._subspace_mode(logits)

        elif self.mode == "combined":
            logits = self._combined_mode(logits)
            """
        
        logits = self.original_lm_head(hidden_states)
        logits = self._combined_mode(logits)

        return logits

    def _token_mode(self, logits: torch.Tensor) -> torch.Tensor:
        if self.pos_token_ids:
            logits[:, :, self.pos_token_ids] += self.alpha
        if self.neg_token_ids:
            logits[:, :, self.neg_token_ids] -= self.alpha
        return logits

    def _subspace_mode(self, logits: torch.Tensor) -> torch.Tensor:
        W = self.original_lm_head.weight   # vocab_size × d
        W_norm = F.normalize(W.float(), dim=-1)

        if self.V is not None:
            V = self.V.to(W.device, dtype=W_norm.dtype)
            score_pos = (( W_norm @ V) ** 2).sum(dim=-1)   # vocab_size
            logits = logits + self.alpha * score_pos.to(logits.dtype)

        if self.V_neg is not None:
            V_neg = self.V_neg.to(W.device, dtype=W_norm.dtype)
            score_neg = ((W_norm @ V_neg) ** 2).sum(dim=-1)
            logits = logits - self.alpha * score_neg.to(logits.dtype)

        return logits

    def _combined_mode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Step 1 — MARINE convex combination:
            log_p_original = log_softmax(logits)
            log_p_guided   = guidance_logits  (set externally)
            log_p_marine   = γ × log_p_guided + (1-γ) × log_p_original

        Step 2 — SVD subspace correction on top of MARINE output:
            logits_final = log_p_marine + α × score_pos - α × score_neg
        """
        # ── Step 1: MARINE ───────────────────────────────────────────────
        log_p_original = F.log_softmax(logits, dim=-1)   # B × T × vocab_size

        if self.guidance_logits is not None:
            # guidance_logits: B × vocab_size → expand to B × T × vocab_size
            g = self.guidance_logits.unsqueeze(1).expand_as(log_p_original)
            g = g.to(log_p_original.dtype)

            log_p_marine = (
                self.gamma * g
                + (1 - self.gamma) * log_p_original
            )
        else:
            # no guidance available — fall back to original
            log_p_marine = log_p_original

        """
        # ── Step 2: SVD subspace correction ─────────────────────────────
        W      = self.original_lm_head.weight          # vocab_size × d
        W_norm = F.normalize(W.float(), dim=-1)

        if self.V is not None:
            V = self.V.to(W.device, dtype=W_norm.dtype)
            score_pos = ((W_norm @ V) ** 2).sum(dim=-1)       # vocab_size
            log_p_marine = (
                log_p_marine
                + self.alpha * score_pos.to(log_p_marine.dtype)
            )

        if self.V_neg is not None:
            V_neg = self.V_neg.to(W.device, dtype=W_norm.dtype)
            score_neg = ((W_norm @ V_neg) ** 2).sum(dim=-1)   # vocab_size
            log_p_marine = (
                log_p_marine
                - self.alpha * score_neg.to(log_p_marine.dtype)
            )
        """

        return log_p_marine