def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    # - A * log_p_theta
    return -raw_rewards_or_advantages * policy_log_probs