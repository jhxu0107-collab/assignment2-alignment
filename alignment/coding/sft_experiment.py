import json
import random
import torch
import wandb
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from alignment.coding.prompt_and_output_tokenization import tokenize_prompt_and_output
from alignment.coding.response_log_probs import get_response_log_probs
from alignment.coding.microbatch_train_step import sft_microbatch_train_step
from alignment.coding.log_generation import log_generations
from alignment.drgrpo_grader import r1_zero_reward_fn, extract_answer
import json

#load data
def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

MODEL_PATH = "models/Qwen2.5-Math-1.5B"
POLICY_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:0"

# copy paste from the assignment
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

# define experiment
def run_sft(run_name, train_data, val_prompts, val_answers, llm, tokenizer, reward_fn,
            sampling_params, lr, microbatch_size, grad_accum_steps, n_sft_steps, eval_every):
    # wandb run  for each experiment
    wandb.init(project="sft", name=run_name, reinit=True)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # load model from huggingface
    model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model.train()
    # load adamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # zero gradiant
    optimizer.zero_grad()
    # initialize weight update
    step = 0
    # initialize minibatch loss update
    step_loss = 0

    # gradient accumulation
    for idx in range(n_sft_steps * grad_accum_steps):
        # each microbatch has microbatch_size data, and random sample
        microbatch = random.sample(train_data, microbatch_size)

        # tokenization
        tok = tokenize_prompt_and_output(
            [ex["prompt"] for ex in microbatch], [ex["response"] for ex in microbatch], tokenizer,
        )
        # take input id and labels
        input_ids = tok["input_ids"].to(POLICY_DEVICE)
        labels = tok["labels"].to(POLICY_DEVICE)
        response_mask = tok["response_mask"].to(POLICY_DEVICE)

        ###### foward pass and backward
        # get the label logits
        log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
        # use output to compute loss
        loss, _ = sft_microbatch_train_step(log_probs, response_mask, grad_accum_steps)
        # take sum within accumulation steps
        step_loss += loss.item()

        # update weights
        if (idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update weights every `gradient_accumulation_steps` batches.
            optimizer.step()
            # Zero gradients every `gradient_accumulation_steps` batches.
            optimizer.zero_grad()
            # record
            wandb.log({"train/loss": step_loss, "train_step": step})
            step_loss = 0 # zero loss

            # update step, add policy update to vllm
            if step % eval_every == 0:
                load_policy_into_vllm_instance(model, llm)
                records, summary = log_generations(llm, model, tokenizer, val_prompts, val_answers, reward_fn,
                                                   sampling_params,
                                                   )
                model.train()
                # evaluate accuracy
                total = 0
                for r in records:
                    total += r["reward"]["answer_reward"]
                accuracy = total / len(records)

                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval/avg_token_entropy": summary["avg_token_entropy"],
                    "eval/avg_response_length": summary["avg_response_length"],
                    "eval_step": step,
                })
            step += 1 # step +1

    wandb.finish()

# args
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_examples", type=int, default=-1)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--n_sft_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--n_val", type=int, default=200)
    args = parser.parse_args()
    random.seed(0107)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = init_vllm(MODEL_PATH, VLLM_DEVICE, seed=0)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    sft = read_jsonl("data/sft.jsonl")
    train_data = sft if args.n_examples == -1 else sft[:args.n_examples]
    reward_fn = r1_zero_reward_fn

    # prompt template
    with open("alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # validation data
    val = read_jsonl("data/MATH/validation.jsonl")
    val_prompts, val_answers = [], []
    for ex in val:
        answer = extract_answer(ex["solution"])
        if answer is None:  # skip, if no boxed answer in the solution
            continue
        val_prompts.append(prompt_template.format(question=ex["problem"]))
        val_answers.append(answer)
    val_prompts, val_answers = val_prompts[:args.n_val], val_answers[:args.n_val]

    reward_fn = r1_zero_reward_fn
    run_name = f"sft_n{len(train_data)}_lr{args.lr}"

    run_sft(run_name, train_data, val_prompts, val_answers, llm, tokenizer, reward_fn, sampling_params,
            lr=args.lr, microbatch_size=args.microbatch_size, grad_accum_steps=args.grad_accum_steps,
            n_sft_steps=args.n_sft_steps, eval_every=args.eval_every)



