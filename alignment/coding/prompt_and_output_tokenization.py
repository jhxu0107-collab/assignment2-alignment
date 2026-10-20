from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """Tokenize the prompt and output strings, and construct a mask that
    is 1 for the response tokens and 0 for other tokens (prompt or padding)."""
    # initialization
    all_ids, all_masks = [],[]
    # for each line in stf, take the same steps:
    for prompt, response in zip(prompt_strs, output_strs):
        # get the token id of prompt and answer
        prompt_token_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        response_token_ids = tokenizer(response, add_special_tokens=False)['input_ids']

        # concatenate the token id together
        ids = prompt_token_ids + response_token_ids

        # mask response token as 1, otherwise 0
        mask = [0] * len(prompt_token_ids) + [1] * len(response_token_ids)

        # add
        all_ids.append(ids)
        all_masks.append(mask)


    # using max length padding
    max_len = max(len(ids) for ids in all_ids)
    # get pad_id
    pad_id = tokenizer.pad_token_id
    # padding
    for ids, mask in zip(all_ids, all_masks):
        if len(ids) < max_len:
            n_pad = max_len - len(ids) # get the # of padding
            ids.extend([pad_id] * n_pad) # repeat pad_id n_pad times
            mask.extend([0] * n_pad)  # mask the padding as 0


    # save as tensor
    all_masks = torch.tensor(all_masks, dtype=torch.long)
    all_ids = torch.tensor(all_ids, dtype=torch.long)

    # shift by one position
    return {
        "input_ids": all_ids[:, :-1],
        "labels": all_ids[:, 1:],
        "response_mask": all_masks[:, 1:],
    }















