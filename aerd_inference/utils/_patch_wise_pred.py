from torch import nn, Tensor
from typing import Union


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


class PatchwisePredictor(nn.Module):
    _MASK_TOKEN = 0
    def __init__(
        self,
        latent_proj,
        proj_over: Union['mean_spec', 'all_spec'] | None = 'mean_spec',
        num_spec_patches: int | None = None,
        apply_mask: bool = False,
    ):
        super().__init__()
        self.latent_proj = latent_proj
        self.proj_over = proj_over
        self.apply_mask = apply_mask

        if proj_over not in ['mean_spec', 'all_spec', None]:
            raise NotImplementedError(f'proj_over = {self.proj_over} not implemented')

    def forward(
        self,
        Z,
        attn_mask=None
    ):
        B, T, M, E = Z.shape
        if self.apply_mask and attn_mask is not None and attn_mask.any():
            Z = Z.reshape(B, -1, E)
            # Require clone for backprop
            Z = Z.clone()
            Z[attn_mask] = self._MASK_TOKEN
            Z = Z.reshape(B, T, M, E)
        # Z : (B, T, M, E)

        if self.proj_over == 'all_spec':
            Z = Z.reshape(B, T, -1)
            # (B, T, M*E)

        # Project from latent embedding space to target space
        logits = self.latent_proj(Z) # E -> C
        # (B, T, M, C)

        if self.proj_over == 'mean_spec':
            if attn_mask is not None:
                attn_mask = attn_mask.reshape(B, T, M)
            logits = mean_with_attn(logits, attn_mask, dim=-2)
            # (B, T, C)

        return logits

    def extra_repr(self):
        return f'proj_over={self.proj_over}'