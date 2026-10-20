import torch
def masked_mean(
tensor: torch.Tensor,
mask: torch.Tensor,
dim: int | None= None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_tensor = tensor * mask
    # if dim is none, calculate the mean over all element
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    # mean over sequence/ any dim
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)