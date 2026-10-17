# load library
import typer
import random
import torch
import wandb
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment.coding.group_normalization import compute_group_normalized_rewards
from alignment.coding.drgrpo_grader import r1_zero_reward_fn
from alignment.coding.zero_shot import load_math_examples, evaluate_vllm
from alignment.coding.microbatch.train import microbatch_train

def grpo_train_loop(run_name, model, optimizer, llm, tokenizer,
                                train_examples, val_prompts, val_answers, reward_fn,
                                grpo_sampling_params, eval_sampling_params,
                                n_grpo_steps, grpo_batch_size, epochs,
                                lr, microbatch_size, gradient_accumulation_steps):
    # wandb run  for each experiment
    wandb.init(project="grpo", name=f"{loss_type}_lr{learning_rate}",
    config=dict(
        loss_type=loss_type,
        learning_rate=learning_rate,
        group_size=group_size,
        rollout_batch_size=rollout_batch_size,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        use_std_normalization=use_std_normalization,
    ),
)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("grpo/*", step_metric="grpo_step")

    step = 0

    for grpo_steps in range(n_grpo_steps):
        # sample a batch of question
        batch = random.sample(train_examples, grpo_batch_size)

        #  load current into VllM
        load_policy_into_vllm_instance(model, llm)

        # G outputs per question, scored by the reward function
        # results layout: [input, response]
        results, metrics = evaluate_vllm(llm, reward_fn, batch, grpo_sampling_params)
                             # llm  # reward function # examples # grpo_sampling_params (g=G)

        # compute advantage with group normalization
        raw_rewards, advantages = compute_group_normalized_rewards(reward_fn, rollout_responses,repeated_ground_truths,
                group_size, advantage_eps, normalize_by_std
        )

        # initialize minibatch loss update
        step_loss = 0
        with torch.no_grad():
            # tokenization
            tok = tokenize_prompt_and_output(
                [ex["prompt"] for ex in results], [ex["response"] for ex in results], tokenizer,
            )
            # take input id and labels
            input_ids = tok["input_ids"].to(POLICY_DEVICE)
            labels = tok["labels"].to(POLICY_DEVICE)
            response_mask = tok["response_mask"].to(POLICY_DEVICE)

            # old_log_prob for response
            old_log_probs = get_response_log_probs(input_ids, labels, model)
            old_log_probs = old_log_probs["log_probs"]
        n_microbatches = len(results) // microbatch_size # how many microbatches a batch have
        for idx in range(n_train_steps_per_rollout_batch * gradient_accumulation_steps):
            # take the %
            mb_idx = idx % n_microbatches
            # slice microbatch
            # start point 1 * microbatch_size
            start = mb_idx * microbatch_size
            # start + micromatch_size
            end = start + microbatch_size
            microbatch = input_ids[start:end]
            labels_micro = labels[start:end]
            response_mask_micro = response_mask[start:end]
            old_log_probs_micro = old_log_probs[start:end]
            advantages_micro = advantages[start:end]
            raw_rewards_micro = raw_rewards[start:end]

            # get policy log
            policy_log_probs = get_response_log_probs(model, microbatch, labels_micro)
            policy_log_probs = policy_log_probs["log_probs"]

            loss = microbatch_train(policy_log_probs, raw_rewards_micro, advantages_micro, loss_type,
                                    cliprange, old_log_probs_micro,
                                    response_mask_micro, gradient_accumulation_steps)
            # accumulate
            step_loss += loss.item()
            wandb.log({"train/loss": step_loss, "train_step": step_offset + step})
            step_loss = 0

            if (idx + 1) % gradient_accumulation_steps == 0:
                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # update
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval/avg_token_entropy": summary["avg_token_entropy"],
                    "eval/avg_response_length": summary["avg_response_length"],
                    "eval_step": step_offset + step,
                })
                step += 1  # step +1

            if manage_wandb:
                wandb.finish()
    return model, optimizer, step_offset + step













        wandb.log({
            "train/loss": loss_value,
            "train/grad_norm": grad_norm,
            "train/token_entropy": entropy_value,
            "train/clip_fraction": clip_fraction,
            "train/reward": reward_mean,
            "train/format_reward": format_mean,
            "train/answer_reward": answer_mean,
        }, step=global_step)

    wandb.finish()
    return model


# args
def main(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,          # As in EI, disallow empty string responses
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,     # On-policy
    train_batch_size: int = 256,           # On-policy
    gradient_accumulation_steps: int = 128,  # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    prompt_path = "alignment/prompts/r1_zero.prompt",
    train_path: str = "data/MATH/train.jsonl",
    val_path: str = "data/MATH/validation.jsonl",
):
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    # data
    train_examples = load_math_examples(train_path, prompt_path)
    val_examples = load_math_examples(val_path, prompt_path)[:.n_val]
    val_prompts = [ex["prompt"] for ex in val_examples]
    val_answers = [ex["ground_truth"] for ex in val_examples]

    # model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = init_vllm(MODEL_PATH, VLLM_DEVICE, seed=args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(POLICY_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # sampling: G rollouts per training question, one per validation question
    ei_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, min_tokens=sampling_min_tokens,
        n=G, seed=seed,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    run_name = f"grpo_G{G}_ep{epochs}_bs{grpo_batch_size}_lr{grpo.lr}"

    grpo_train_loop(run_name, model, optimizer, llm, tokenizer,
        run_name, model, optimizer, llm, tokenizer,
        train_examples, val_prompts, val_answers, r1_zero_reward_fn,
        ei_sampling_params, eval_sampling_params,
        n_ei_steps=n_grpo_steps, ei_batch_size=grpo_batch_size, epochs=epochs,
        lr=lr, microbatch_size=microbatch_size, grad_accum_steps=gradient_accumulation_stepssteps,
    )