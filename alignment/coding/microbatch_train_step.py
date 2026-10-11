import torch
from alignment.coding.masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # batch size
    batch_size = policy_log_probs.size(0)
    # masked sum over response tokens, averaged over the batch
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant) / batch_size
    # scale for gradient accumulation
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, {}