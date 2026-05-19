import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDGuidedLMHead(nn.Module):
    def __init__(self, original_lm_head, alpha=1.0, top_k=None):
        """
        Wraps the original LM head.
        Before projecting to vocab, modifies hidden state u as:
            u_new = u + alpha * VVᵀu
        where V comes from SVD of the object/caption representation matrix.

        Args:
            original_lm_head: the nn.Linear lm_head from the LLM
            alpha: amplification strength
            top_k: how many singular vectors to keep. None = keep all
        """
        super().__init__()
        self.original_lm_head = original_lm_head
        self.alpha = alpha
        self.top_k = top_k
        self.VVT = None   # d × d projection matrix, set by precompute()
        self.V = None 

    def precompute(self, representation_matrix: torch.Tensor):
        """
        Args:
            representation_matrix: M x d tensor
                                   M = number of objects (or objects + captions)
                                   d = LLM hidden dimension (4096 for 7B)
        """
        X = representation_matrix.float()

        # normalize rows so no single object dominates
        X = F.normalize(X, dim=-1)

        # SVD: X = U S Vᵀ
        # Vh shape: k × d  (torch returns Vᵀ)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        # optionally keep only top-k singular vectors
        if self.top_k is not None:
            Vh = Vh[:self.top_k, :]

        # V: d × k
        self.V = Vh.T
        self.V, _ = torch.linalg.qr(self.V)

        # projection matrix VVᵀ: d × d
        #device = next(self.original_lm_head.parameters()).device
        #self.VVT = (V @ V.T).to(device).half()   # half() to match model dtype

        #print(f"[SVDGuidedLMHead] VVᵀ computed. "
              #f"Representation matrix: {X.shape}, "
              #f"Singular vectors kept: {V.shape[1]}")

    def reset(self):
        """Call between images."""
        self.VVT = None
        self.V = None

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: B x T x d
        Returns:
            logits: B x T x vocab_size
        """
        if self.V is not None:

            # match device/dtype dynamically
            V = self.V.to(
                hidden_states.device,
                dtype=hidden_states.dtype
            )

            # hidden_states: B x T x d
            # V: d x k

            coeffs = hidden_states @ V          # B x T x k

            amplification = coeffs @ V.T        # B x T x d

            hidden_states = (
                hidden_states
                + self.alpha * amplification
            )

        return self.original_lm_head(hidden_states)