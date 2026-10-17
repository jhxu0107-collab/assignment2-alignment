from alignment.coding.compute_naive_policy_loss import compute_naive_policy_gradient_loss
from alignment.coding.grpo_clip.loss import compute_grpo_clip_loss
def compute_policy_gradient_loss(
policy_log_probs: torch.Tensor,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None= None,
advantages: torch.Tensor | None= None,
old_log_probs: torch.Tensor | None= None,
cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # with no baseline
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards cannot be None"
        loss = compute_naive_policy_gradient_loss(policy_log_probs, raw_rewards)
        return loss, {"loss": loss}

    # - baseline
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages cannot be None"
        loss = compute_naive_policy_gradient_loss(policy_log_probs, advantages)
        return loss, {"loss": loss}

    # grpo_clip
    elif loss_type == "grpo_clip":
        assert advantages is not None, "grpo_clip requires advantages"
        assert old_log_probs is not None, "grpo_clip requires old_log_probs"
        assert cliprange is not None, "grpo_clip requires cliprange"
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        # ratio of clipped
        metadata["clip_fraction"] = metadata["is_clipped"].float().mean()
        return loss, metadata

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")




