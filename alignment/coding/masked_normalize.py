import torch
def masked_normalize(
tensor: torch.Tensor,
mask: torch.Tensor,
normalize_constant: float,
dim: int | None= None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        #If None, sum over all dimensions.
        total = masked_tensor.sum()
    else:
        total = masked_tensor.sum(dim=dim)  # the dimension to sum along before normalization.
    return total / normalize_constant
