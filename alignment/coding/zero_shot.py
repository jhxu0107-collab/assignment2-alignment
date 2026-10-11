# evaluate zero-shot performance
from vllm import LLM, SamplingParams

def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
eval_sampling_params: SamplingParams
) -> None:
"""
Evaluate a language model on a list of prompts,
compute evaluation metrics, and serialize results to disk.
"""
# (1) data loader
example = []
validation = "/Users/a75700/Desktop/ml_cse/assignment/assignment2-alignment/data/MATH/validation.jsonl"
with open(validation) as f:
    for line in f:
        # skip blank line
        if not line.strip():
            continue
        # load line and make variables
        dataset = json.loads(line)
        problem = dataset["problem"]
        solution = dataset["solution"]
        level= dataset["level"]
        problem_type = dataset["problem_type"]
        dataset['id'] = len(examples)
        # save each record
        examples.append(dataset)

        # (2) format val as prompt
        prompt_template = readline('/Users/a75700/Desktop/ml_cse/assignment/assignment2-alignment/alignment/prompts/r1_zero.prompt')
        prompt_template = prompt_template.strip('\n')
        # replace question with val
        example['prompt'] = prompt_template.replace("{question}", problem)
        examples.append(example)

        # (3) generate output for each example
        output = []
        for i, sample in enumerate(dataset["samples"]):
            prompt = sample["prompt"]
            # Create a sampling params object, stopping generation on newline.
            sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])
            # Create an LLM.
            llm = LLM(model=<path to model>)
            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.
            outputs = llm.generate(prompts, sampling_params)

            # Print the outputs
            for output in outputs:
                prompt = output.prompt
            generated_text = output.outputs[0].text
            # save the output
            output = [
                dict(idx = idx, generated_text = generated_text)
                for idx in range(len(prompt))
            ]

        # (4) evaluation
        for i, sample in enumerate(dataset["samples"]):
            if output['generated_text'] = examples['solution']
                output['evaluation'] = True
            else:
                output['evaluation'] = False
        metric =



            # parse gt
            for sample in samples:
                sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
            params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

            scores = []
            timeout_cnt = 0

            with ProcessPool(max_workers=1) as pool:
                future = pool.map(math_equal_process, params, timeout=3)
                iterator = future.result()
                with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
                    while True:
                        try:
                            result = next(iterator)
                            scores.append(result)
                        except StopIteration:
                            break
                        except TimeoutError as error:
                            print(error)
                            scores.append(False)
                            timeout_cnt += 1
                        except Exception as error:
                            print(error.traceback)
                            exit()
                        progress_bar.update(1)

            idx = 0
            score_mat = []
            for sample in samples:
                sample['score'] = scores[idx: idx + len(sample['pred'])]
                assert len(sample['score']) == len(sample['pred'])
                score_mat.append(sample['score'])
                idx += len(sample['pred'])

            max_len = max([len(s) for s in score_mat])

            for i, s in enumerate(score_mat):
                if len(s) < max_len:
                    score_mat[i] = s + [s[-1]] * (max_len - len(s))  # pad

            # output mean of each column of scores
            col_means = np.array(score_mat).mean(axis=0)
            mean_score = list(np.round(col_means * 100, decimals=1))

            result_json = {
                "num_samples": len(samples),
                "num_scores": len(scores),
                "timeout_samples": timeout_cnt,
                "empty_samples": len([s for s in samples if not s['pred'][-1]]),
                "acc": mean_score[0]
            }

            # each type score
            if "type" in samples[0]:
                type_scores = {}
                for sample in samples:
                    if sample['type'] not in type_scores:
                        type_scores[sample['type']] = []
                    type_scores[sample['type']].append(sample['score'][-1])
                type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
                type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
                result_json['type_acc'] = type_scores

            print(result_json)
            return samples, result_json


        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--data_name", type=str, default="math")
            parser.add_argument("--prompt_type", type=str, default="tool-integrated")
            parser.add_argument("--file_path", type=str, default=None, required=True)
            parser.add_argument("--max_num_samples", type=int, default=None)
            parser.add_argument("--execute", action="store_true")
            args = parser.parse_args()
            return args


        if __name__ == "__main__":
            args = parse_args()
            evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
                     max_num_samples=args.max_num_samples, execute=args.execute)


