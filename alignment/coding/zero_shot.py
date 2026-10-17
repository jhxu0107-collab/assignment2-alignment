import json
# evaluate zero-shot performance
from vllm import LLM, SamplingParams
from alignment.drgrpo_grader import r1_zero_reward_fn, extract_answer

# load data and format as prompt
def load_math_examples(data_path, prompt_path):
    """(1) Load MATH examples and (2) format them as prompts."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():    # skip blank line
                continue
            ex = json.loads(line)
            # extract solution
            answer = extract_answer(ex["solution"])
            if answer is None:          # no boxed answer -> cannot grade
                continue
            ex["id"] = len(examples)
            ex["ground_truth"] = answer
            ex["prompt"] = prompt_template.format(question=ex["problem"])
            examples.append(ex)
    return examples

# generate response and evaluate
def evaluate_vllm(llm, reward_fn, examples, sampling_params, output_path=None):
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # (3) generate output for all examples
    results = []
    prompts = [ex["prompt"] for ex in examples]
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # (4) evaluate
    for ex, out in zip(examples, outputs):
        for candidate in out.outputs:  #  1 or g answers # changed for expert iteration
            response = candidate.text
            reward = reward_fn(response, ex["ground_truth"])
            results.append({**ex, "response": response, **reward})

    n = len(results)
    metrics = {
        "num_examples": n,
        "accuracy": sum(r["answer_reward"] for r in results) / n,
        "format_rate": sum(r["format_reward"] for r in results) / n,
        "format1_answer1": sum(r["format_reward"] == 1 and r["answer_reward"] == 1 for r in results),
        "format1_answer0": sum(r["format_reward"] == 1 and r["answer_reward"] == 0 for r in results),
        "format0_answer0": sum(r["format_reward"] == 0 and r["answer_reward"] == 0 for r in results),
    }
    # (5) serialize examples, generations, and scores
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results, metrics

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    examples = load_math_examples(data_path="data/MATH/validation.jsonl",
                                  prompt_path="alignment/prompts/r1_zero.prompt")
    if limit:
        examples = examples[:limit]
    print(f"Evaluating {len(examples)} examples")

    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    output_path = "zero_shot_results_test.jsonl" if limit else "zero_shot_results.jsonl"
    results, metrics = evaluate_vllm(
        llm, r1_zero_reward_fn, examples, sampling_params, output_path,
    )
    print(metrics)

