from torch import nn, Tensor


def mean_with_attn(x: Tensor, mask: Tensor | None, dim: int) -> Tensor:
    """Compute mean along dimension, ignoring masked positions."""
    if mask is None:
        return x.mean(dim=dim)
    # Clone to avoid modifying input
    x = x.clone()
    # Expand mask to match x dimensions if needed
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    x[mask] = 0
    count = (~mask).sum(dim=dim, keepdim=True).float()
    return x.sum(dim=dim) / count.squeeze(dim)


class HardLabelPredictor(nn.Module):
    _MASK_TOKEN = 0
    def __init__(self, latent_proj, apply_mask=False):
        super().__init__()
        self.latent_proj = latent_proj
        self.apply_mask = apply_mask

    def forward(self, Z, attn_mask=None):
        B, T, M, E = Z.shape
        Z = Z.reshape(B, -1, E)
        # (B, T*M, E)
        if self.apply_mask and attn_mask is not None and attn_mask.any():
            # Require clone for backprop
            Z = Z.clone()
            Z[attn_mask] = self._MASK_TOKEN

        logits = self.latent_proj(Z) # E -> C
        # (B, T*M, C)
        logits = mean_with_attn(logits, attn_mask, dim=-2)
        # (B, C)
        return logits