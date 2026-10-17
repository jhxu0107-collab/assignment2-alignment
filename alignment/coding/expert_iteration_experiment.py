# load library
import argparse
import random
import torch
import wandb
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.coding.zero_shot import load_math_examples, evaluate_vllm
from alignment.coding.sft_experiment import (
    run_sft, init_vllm, load_policy_into_vllm_instance,
    MODEL_PATH, POLICY_DEVICE, VLLM_DEVICE,
)

def expert_iteration_experiment(run_name, model, optimizer, llm, tokenizer,
                                train_examples, val_prompts, val_answers, reward_fn,
                                ei_sampling_params, eval_sampling_params,
                                n_ei_steps, ei_batch_size, epochs,
                                lr, microbatch_size, grad_accum_steps):
    # wandb run  for each experiment
    wandb.init(project="expert_iteration", name=run_name, reinit=True)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    global_step = 0
    for ei_step in range(n_ei_steps):
        # sample a batch of question
        batch = random.sample(train_examples, ei_batch_size)
        # base model
        load_policy_into_vllm_instance(model, llm)
        # G outputs per question, scored by the reward function
        results, metrics = evaluate_vllm(llm, reward_fn, batch, ei_sampling_params)
                             # llm  # reward function # examples # ei_sampling_params (g=G)
        # filter the correct answer
        sft_data = [{'prompt':r['prompt'],'response': r['response']}
                    for r in results if r['answer_reward']==1]
        # record
        wandb.log({
            "ei/rollout_accuracy": metrics["accuracy"],  # accuracy
            "ei/num_sft_examples": len(sft_data), # number of sft examples
            "ei_step": ei_step,  # step
        })

        # ff and backward using correct
        sft = run_sft(train_data=correct)

        # SFT on the correct outputs, continuing from the current model
        # how many weights
        n_sft_steps = max(1, epochs * len(sft_data) // (microbatch_size * grad_accum_steps))
        model, optimizer, global_step = run_sft(
            run_name, sft_data, val_prompts, val_answers, llm, tokenizer, reward_fn,
            eval_sampling_params, lr, microbatch_size, grad_accum_steps, n_sft_steps,
            eval_every=n_sft_steps,  # evaluate only at the end of each EI step
            model=model, optimizer=optimizer, manage_wandb=False, step_offset=global_step,
        )

    wandb.finish()
    return model


# args
import argparse
sampling_min_tokens=4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_examples", type=int, default=-1)
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--n_sft_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # data
    prompt_path = "alignment/prompts/r1_zero.prompt"
    train_examples = load_math_examples("data/MATH/train.jsonl", prompt_path)
    val_examples = load_math_examples("data/MATH/validation.jsonl", prompt_path)[:args.n_val]
    val_prompts = [ex["prompt"] for ex in val_examples]
    val_answers = [ex["ground_truth"] for ex in val_examples]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = init_vllm(MODEL_PATH, VLLM_DEVICE, seed=args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(POLICY_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # sampling: G rollouts per training question, one per validation question
    sampling_min_tokens = 4
    ei_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, min_tokens=sampling_min_tokens,
        n=args.G, seed=args.seed,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    run_name = f"ei_G{args.G}_ep{args.epochs}_bs{args.ei_batch_size}_lr{args.lr}"

    expert_iteration_experiment(
        run_name, model, optimizer, llm, tokenizer,
        train_examples, val_prompts, val_answers, r1_zero_reward_fn,
        ei_sampling_params, eval_sampling_params,
        n_ei_steps=args.n_ei_steps, ei_batch_size=args.ei_batch_size, epochs=args.epochs,
        lr=args.lr, microbatch_size=args.microbatch_size, grad_accum_steps=args.grad_accum_steps,
    )