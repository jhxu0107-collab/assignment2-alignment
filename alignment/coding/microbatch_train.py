from alignment.coding.policy_gradient_wrapper import compute_policy_gradient_loss
from alignment.coding.masked_mean import mased_mean
import torch
from typing import Literal
def grpo_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None= None,
advantages: torch.Tensor | None= None,
old_log_probs: torch.Tensor | None= None,
cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # per token loss
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # take the mean over each sequence
    sequence_loss = masked_mean(per_token_loss, response_mask, dim=1)

    # batch mean
    loss = sequence_loss.mean()

    # gradient scaling
    loss = loss / gradient_accumulation_steps
    # backward
    loss.backward()

    return loss, metadata



