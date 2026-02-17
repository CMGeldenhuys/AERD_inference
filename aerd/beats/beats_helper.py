from functools import lru_cache
from typing import Dict, List, Tuple

import torch
from numpy import floor
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

try:
    from .unilm.beats import BEATs, BEATsConfig

except ImportError as e:
    print(f"BEATs submodule not found, please initialise git submodules: {e}")
    raise e


def build_beats_model(
    weights_path: str | None = None, disable_layerdrop: bool = True, **cfg
) -> BEATs:
    checkpoint = None
    if weights_path:
        assert not cfg
        checkpoint = torch.load(weights_path, weights_only=True)
        cfg = checkpoint["cfg"]

    if isinstance(cfg, dict):
        cfg = BEATsConfig(cfg)
    assert isinstance(cfg, BEATsConfig)
    # Useful when using DDP as layerdrop does not work in the distributed
    # case. Not sure why since the seeds are synced across all nodes but I
    # suspect it has to do with the gradient graph. Since the encoder
    # layers are dropped on a per sample basis not per batch and so some
    # outputs go through all encoders while others don't.
    # See issue: https://github.com/pytorch/pytorch/issues/42793
    # Setting `find_unused_parameters=True`, seems to also solve the issue
    # at the cost of longer training epochs
    if disable_layerdrop:
        cfg.encoder_layerdrop = 0.0  # set prob to 0, no chance of drop out

    model = BEATs(cfg)
    if weights_path:
        assert checkpoint
        model.load_state_dict(checkpoint["model"])

    # disable beats own predictor, only use as feature extractor
    model.predictor = None
    return model


@lru_cache(maxsize=1)
def get_num_spectral_patches(model, num_mel_bins=128):
    # Compute number of spectral patches
    padding = model.patch_embedding.padding
    dilation = model.patch_embedding.dilation
    stride = model.patch_embedding.stride
    kernel_size = model.patch_embedding.kernel_size
    W_out = floor(
        (num_mel_bins + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
        / stride[1]
        + 1
    )
    return int(W_out)


def beats_preprocessing(
    model: BEATs,
    X: torch.Tensor | PackedSequence,  # shape: (batch, audio_len, n_channels, *)
    padding_mask: None | torch.Tensor | List[int] = None,
) -> Tuple[Tensor, Tensor]:
    pad_len = 0
    if isinstance(X, PackedSequence):
        assert padding_mask is None
        X, seq_len = pad_packed_sequence(X, batch_first=True)
        audio_len = X.shape[1]
        # seq_len is the length of each unpadded seq
        # pad_len is the length of padding at the end of the seq
        pad_len = audio_len - seq_len
        # TODO: change seq_len instead, leave not that in AERD this is pad_len not seq_len
    # NOTE: check PackedSequence first as PackedSequence is a subtype of Tuple
    # If multiple args are given by X
    # assume, second arg is pad_len
    elif isinstance(X, tuple) or isinstance(X, list):
        assert padding_mask is None
        X, pad_len = X

    add_batch_dim = X.ndim == 1

    if X.ndim > 2:
        batch_size, audio_len, *_ = X.shape
        # Flatten all dims, except batch
        X = X.flatten(start_dim=1)

    elif add_batch_dim:
        X = X.unsqueeze(0)

    X = X.contiguous()

    batch_size, audio_len = X.shape

    # If pad_len not set,
    # assume no padding
    # i.e. not attention masking
    if isinstance(pad_len, int):
        pad_len = torch.full((batch_size,), fill_value=pad_len, dtype=torch.long)

    fbank = model.preprocess(X)
    if padding_mask is None:
        # Ensure padding mask on same device as x
        # see https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#init-tensors-using-tensor-to-and-register-buffer
        padding_mask = torch.zeros(
            batch_size, audio_len, device=X.device, dtype=torch.bool
        )
        # Mark all samples that correspond to padding tokens
        for i, pl in enumerate(pad_len):
            if pl == 0:
                continue
            padding_mask[i, -pl:] = True

        padding_mask = model.forward_padding_mask(fbank, padding_mask)

    # Add back empty channel dim, expected by backbone/patch embedding
    # fbank = fbank.unsqueeze(1)
    # Remove channel dim
    # padding_mask = padding_mask.squeeze()
    assert isinstance(padding_mask, Tensor)

    if add_batch_dim:
        fbank = fbank.squeeze(0)
        padding_mask = padding_mask.squeeze(0)

    return fbank, padding_mask


def beats_extract_features(
    model: BEATs,
    X: Tensor | Tuple[Tensor, Tensor],
    # return_spectral_patch=False,  # Return spectral patch dim, or take mean
    # return_fbank=False,
    padding_mask: Tensor | None = None,
    *,
    apply_mask: bool = True,
    return_encoder_layer: int | None = None,
    random_feature_mask: bool = False,
    mask_token: float | Tensor | Parameter = 0.0,
) -> Tuple[Tensor, Dict]:
    # Optional return dict
    opt_ret = dict()
    # X contains multiple arguments
    X_is_multi_arg = isinstance(X, tuple) or isinstance(X, list)

    if padding_mask is not None and X_is_multi_arg:
        fbank, _ = X
    elif X_is_multi_arg:
        fbank, padding_mask = X
    else:
        fbank = X
    assert isinstance(fbank, Tensor)

    add_batch_dim = bool(fbank.ndim == 2)
    if add_batch_dim:
        # Add batch dim
        fbank = fbank.unsqueeze(0)
        # Add channel dim
        fbank = fbank.unsqueeze(1)
    elif fbank.ndim == 3:
        # Add channel dim
        fbank = fbank.unsqueeze(1)

    if padding_mask is None:
        padding_mask = torch.zeros(fbank.size(-3), fbank.size(-2))
    assert isinstance(padding_mask, Tensor)

    if add_batch_dim:
        assert padding_mask.ndim == 1
        padding_mask = padding_mask.unsqueeze(0)

    assert (
        fbank.ndim == 4
    ), "fbanks must be of shape: (batch, nchannels, time_dim, mel_dim)"
    assert padding_mask is None or padding_mask.size(0) == fbank.size(
        0
    ), "batch_size must be equal for padding_mask and fbanks"
    assert not padding_mask.all(dim=1).any(), "batch contains sample with just padding"

    batch_size, nchannels, time_dim, mel_dim = fbank.shape

    if random_feature_mask:
        raise NotImplementedError
        feat_mask = feature_mask(
            fbank.size(),
            threshold=self.feature_mask_threshold,
            kernel_size=self.feature_mask_kernel_size,
            device=fbank.device,
        )
        fbank = fbank * feat_mask

    if return_encoder_layer is not None:
        # layer_results -> [(x_i, attn)]
        # x_i : output of current layer fed to next layer
        # attn : attention weights, averaged over heads
        # index 0: is inputs to encoder layer
        Z, attn_mask, encoder_layers = model.forward(
            fbank,
            padding_mask=padding_mask,
            return_encoder_layer=return_encoder_layer,
        )
        if return_encoder_layer is True:
            opt_ret["encoder_layers"] = encoder_layers
        elif isinstance(return_encoder_layer, int):
            opt_ret["encoder_layer"] = return_encoder_layer
            Z, _mask = encoder_layers[return_encoder_layer]
            Z = Z.transpose(0, 1)
            attn_mask = None
        else:
            raise ValueError("Unreachable")
    else:
        Z, attn_mask = model.forward(
            fbank,
            padding_mask=padding_mask,
        )

    assert isinstance(Z, Tensor)
    assert isinstance(attn_mask, Tensor) or attn_mask is None

    apply_mask = bool(apply_mask and (attn_mask is not None) and attn_mask.any())
    if apply_mask:
        # Mask output tokens
        # Note: these tokens contain random info, as they were masked to the encoder
        Z = Z.clone()  # Expensive copy needed for backprop graph
        # Can't modify view inplace: https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308
        # Z = Z * attn_mask
        Z[attn_mask] = mask_token
        # Masked tensor impl: shapes through linear layer end up breaking
        # Z = masked_tensor(Z, attn_mask.unsqueeze(-1).expand_as(Z))

    num_spectral_patches = get_num_spectral_patches(model)
    # TODO: get from config
    num_features = 768
    # Reshape the hidden embedding tokens Z
    # (batch_size, num_tokens_temporal, num_tokens_spectral, num_features)
    # TODO: hard codeded feature dim
    Z = Z.reshape(batch_size, -1, num_spectral_patches, num_features)

    # Remove batch dim if one was inserted
    if add_batch_dim:
        Z = Z.squeeze(0)
        attn_mask = attn_mask.squeeze(0) if attn_mask is not None else None

    # If mask was applied, store attn_mask
    if apply_mask:
        opt_ret["attn_mask"] = attn_mask

    return Z, opt_ret
