import os
import numpy as np
import torch


def patchstack(patches, border=2, padvalue=1.0):
    """
    Stack field of patches into one large image.

    Inputs
    ------
    patches    Tensor of shape [..., R, R, H', W']
    border     Space (in pixels) between neighboring patches (integer)
    padvalue   Value to fill space with

    Outputs
    -------
               Tensor of shape [..., (R+border)*H'-border, (R+border)*W'-border], containing stacked patches

    """
    assert border % 2 == 0, f"border must be even (but got {border})"

    # Pad 3rd and 4th to last dimensions with border//2 pixels valued `padvalue`.
    padamt = (0, 0, 0, 0, border//2, border//2, border//2, border//2)
    padded = torch.nn.functional.pad(patches, padamt, value=padvalue).detach().cpu()

    permutation = list(range(len(patches.shape)))
    permutation[-4] = -2
    permutation[-3] = -4
    permutation[-2] = -1
    permutation[-1] = -3

    new_shape = list(padded.shape[:-2])
    new_shape[-2] *= padded.shape[-2]
    new_shape[-1] *= padded.shape[-1]

    output = padded.permute(permutation).contiguous().view(new_shape)
    
    return output


def tile(a, dim, n_tile):
    """
    Tile tensor a along dimension `dim` with `n_tile` repeats.

    Written by Edouard360:
    https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

