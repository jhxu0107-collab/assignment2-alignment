import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension)."""
    # discrete distribution p(x)
    logp = torch.log_softmax(logits, dim=-1) # stable normalization
    p = logp.exp() # p(x)
    entropy = -(p * logp).sum(dim=-1)
    return entropy

