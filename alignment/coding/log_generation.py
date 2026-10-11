import torch
from alignment.coding.prompt_and_output_tokenization import tokenize_prompt_and_output
from alignment.coding.response_log_probs import get_response_log_probs
from alignment.coding.masked_normalize import masked_normalize

def log_generations(llm, model, tokenizer, prompts, ground_truths, reward_fn, sampling_params):
    """Generate responses for the given prompts and log per-example info and summary stats."""
    #  generate response by llm
    outputs = llm.generate(prompts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    model.eval()
    records = []

    for prompt, response, gt in zip(prompts, responses, ground_truths):
        # get reward
        reward = reward_fn(response, gt)

        # average token
        tok = tokenize_prompt_and_output([prompt], [response], tokenizer)
        input_ids = tok["input_ids"].to(model.device)
        labels = tok["labels"].to(model.device)
        mask = tok["response_mask"].to(model.device)
        length = int(mask.sum().item())

        if length > 0:
            with torch.no_grad():
                out = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            avg_entropy = masked_normalize(out["token_entropy"].float(), mask, length).item()
        else:
            avg_entropy = 0.0

        # save
        records.append({
            "prompt": prompt,               # record prompt
            "response": response,                # record response
            "ground_truth": gt,        # ground truth
            "reward": reward,           # reward
            "avg_token_entropy": avg_entropy,  # entropy
            "response_length": length, # response length
        })

    # summary
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    correct = [r for r in records if r["reward"]["answer_reward"] == 1]
    incorrect = [r for r in records if r["reward"]["answer_reward"] != 1]

    summary = {
        "avg_token_entropy": mean([r["avg_token_entropy"] for r in records]),
        # the average response length
        "avg_response_length": mean([r["response_length"] for r in records]),
        # average response length for correct responses
        "avg_length_correct": mean([r["response_length"] for r in correct]),
        # average response length for incorrect responses.
        "avg_length_incorrect": mean([r["response_length"] for r in incorrect]),
    }

    return records, summary

