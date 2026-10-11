import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from alignment.coding.per_token_entropy import compute_entropy

def get_response_log_probs(
model: PreTrainedModel,
input_ids: torch.Tensor,
labels: torch.Tensor,
return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits # next token logits
    logp = F.log_softmax(logits,dim=-1)
    # get the log probability of y
    log_probs = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": log_probs}
    # return token entropy
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)  # (batch, seq_len)
    return result
