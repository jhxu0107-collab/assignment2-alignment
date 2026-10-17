


def compute_grpo_clip_loss(
advantages: torch.Tensor,
policy_log_probs: torch.Tensor,
old_log_probs: torch.Tensor,
cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # calculate ratio
    ratio = torch.exp(policy_log_probs - old_log_probs)
     # advantage is a scalar -> broadcast over sequence length
    unclipped = ratio * advantages
    # ratio > 1.2, clip
    # ratio < 1.2 clip
    clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.minimum(unclipped, clipped)
    # which tokens are clipped
    is_clipped = clipped < unclipped

    return loss, {"is_clipped": is_clipped}

