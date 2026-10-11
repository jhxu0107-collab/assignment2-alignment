import json
import re
from alignment.drgrpo_grader import r1_zero_reward_fn, extract_answer
answer = {}
# use gt in math as gt
with open("data/MATH/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        # extract answer
        answer = extract_answer(ex["solution"])
        if answer is not None:
            answers[norm(ex["problem"])] = answer

# get the prefix and suffix
with open("alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
    pre, suf = f.read().split("{question}")

example = []
with open("data/sft.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        question = ex["prompt"].removeprefix(pre).removesuffix(suf)
        gt = answers.get(norm(question))  # None if the question is not in MATH train
        ex["ground_truth"] = gt
        if gt is None:
            ex["correct"] = False  # no MATH ground truth treated as incorrect
        else:
            ex["correct"] = r1_zero_reward_fn(ex["response"], gt)["answer_reward"] == 1
        examples.append(ex)

with open("data/sft_with_gt.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

matched = sum(ex["ground_truth"] is not None for ex in examples)
correct = sum(ex["correct"] for ex in examples)
print("total:", len(examples))
print("found in MATH train:", matched)
print("correct (kept):", correct)
