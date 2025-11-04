from parse_dataset import load_json_lines
import json
from config import *
from parse_ast import *
import re
from call_llm import gpt_4o_mini_inference
def gen_developer_prompt(function_calls: list, prompt_passing_in_english: bool):
    function_calls_json = json.dumps(function_calls, ensure_ascii=False, indent=2)
    passing_in_english_prompt = " Pass in all parameters in function calls in English." if prompt_passing_in_english else ""
    return f'''
You are an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.{passing_in_english_prompt}
            
At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\"s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
            
Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''

def gen_input_messages(developer_prompt: str, user_question: str) -> dict:
    developer_message = {"role": "developer", "content": developer_prompt}
    user_message = {"role": "user", "content": user_question}
    return [developer_message, user_message]

def inference(test_entry: dict):
    # print(test_entry)
    functions = test_entry['function']
    developer_prompt = gen_developer_prompt(
        function_calls=functions,
        prompt_passing_in_english=True
    )
    user_question = test_entry["question"][0][0]['content']
    input_messages = gen_input_messages(
        developer_prompt=developer_prompt,
        user_question=user_question
    )
    # print("Prompt dict:")
    # print(input_messages[0]['content'])
    # print(input_messages[1]['content'])
    # to do: call the LLM API with prompt_dict

    result = gpt_4o_mini_inference(input_messages)

    result_to_write = {
        "id": test_entry["id"],
        "result": result
    }
    return result_to_write



# Run inference

for config in configs:
    print(f"Processing config: {config}")
    match config.translate_info:
        case Translated(language=lang, translate_mode=mode):
            match lang:
                case Language.CHINESE:
                    language_postfix = "_zh"
                case Language.HINDI:
                    language_postfix = "_hi"
            match mode:
                case TranslateMode.DATASET_FULLY_TRANSLATED_PROMPT_DEFAULT:
                    translate_dataset_prefix = "_full"
                    translate_mode_prefix = "_d" # default
                case TranslateMode.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE:
                    translate_dataset_prefix = "_full"
                    translate_mode_prefix = "_pt"  # prompt translate
                case TranslateMode.DATASET_PARTIALLY_TRANSLATED:
                    translate_dataset_prefix = "_partial"
                    translate_mode_prefix = "_par" # partial
        case NotTranslated():
            language_postfix = ""
            translate_dataset_prefix = ""
            translate_mode_prefix = ""
    match config.add_noise_mode:
        case AddNoiseMode.NO_NOISE:
            noise_postfix = ""
        case AddNoiseMode.ADD_NOISE:
            noise_postfix = "_noisy"
    dataset_path = f"dataset/BFCL_v4_multiple{language_postfix}{translate_dataset_prefix}{noise_postfix}.json"
    ground_truth_path = f"dataset/possible_answer/BFCL_v4_multiple.json"
    inference_result_path = f"result/inference/BFCL_v4_multiple{language_postfix}{translate_mode_prefix}{noise_postfix}.json"
    evaluation_result_path = f"result/evaluation/BFCL_v4_multiple{language_postfix}{translate_mode_prefix}{noise_postfix}.json"
    score_path = f"result/score/BFCL_v4_multiple{language_postfix}{translate_mode_prefix}{noise_postfix}.json"
    with open(dataset_path, 'r', encoding='utf-8') as f_dataset:
        test_cases = load_json_lines(f_dataset)
    # test_cases = test_cases[:1]
    # open or create the inference result file
    if requires_inference:
        inference_results = []
        try:
            with open(inference_result_path, 'r', encoding='utf-8') as f_inference_out:
                if f_inference_out.readable():
                    # read all lines and parse as list of dict
                    for line in f_inference_out:
                        inference_results.append(json.loads(line))
        except FileNotFoundError:
            print(f"Inference result file {inference_result_path} not found. It will be created.")
        existing_inference_ids = {entry['id'] for entry in inference_results}
        printed_warning = False
        with open(inference_result_path, 'w', encoding='utf-8') as f_inference_out:
            for i, case in enumerate(test_cases):
                if case['id'] in existing_inference_ids:
                    if not printed_warning:
                        print(f"Warning: some test cases already exist in inference result file. Skipping.")
                        printed_warning = True
                    continue
                # the actual inference
                print(f"Inferencing case id {case['id']}, question: {case['question'][0][0]['content']}")
                result = inference(case)
                print("Answer: ", result["result"])
                inference_results.append(result)
                f_inference_out.seek(0)
                f_inference_out.truncate()
                for result in inference_results:
                    f_inference_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_inference_out.flush()                
            # rewrite all results to the file
            inference_results = sorted(inference_results, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
            f_inference_out.seek(0)
            f_inference_out.truncate()
            for result in inference_results:
                f_inference_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_inference_out.flush()
    if requires_evaluation:
        evaluation_results = []
        try:
            with open(evaluation_result_path, 'r', encoding='utf-8') as f_evaluation_out:
                if f_evaluation_out.readable():
                    # read all lines and parse as list of dict
                    for line in f_evaluation_out:
                        evaluation_results.append(json.loads(line))
        except FileNotFoundError:
            print(f"Evaluation result file {evaluation_result_path} not found. It will be created.")
        existing_evaluation_ids = {entry['id'] for entry in evaluation_results}
        printed_warning = False
        with open(inference_result_path, 'r', encoding='utf-8') as f_inference_in, \
                open(evaluation_result_path, 'w', encoding='utf-8') as f_evaluation_out, \
                open(dataset_path, 'r', encoding='utf-8') as f_dataset_in, \
                open(ground_truth_path, 'r', encoding='utf-8') as f_ground_truth_in:
            if not f_inference_in.readable():
                print(f"Error: inference result file {inference_result_path} is not readable.")
                continue
            inference_results = []
            ground_truths = []
            dataset = []
            for line in f_dataset_in:
                test_entry = json.loads(line)
                dataset.append(test_entry)
            for line in f_inference_in:
                inference_results.append(json.loads(line))
            for line in f_ground_truth_in:
                ground_truths.append(json.loads(line))
            for (inference_line, ground_truth_line, dataset_line) in zip(inference_results, ground_truths, dataset):
                id = inference_line["id"]
                if id in existing_evaluation_ids:
                    if not printed_warning:
                        print(f"Warning: some test cases already exist in evaluation result file. Skipping.")
                        printed_warning = True
                    continue
                assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
                assert id == dataset_line["id"], f"Mismatch in IDs: {id} vs {dataset_line['id']}"
                inference_result = inference_line["result"]
                ground_truth = ground_truth_line["ground_truth"]
                func_description = dataset_line['function']
                # print("func_description: ", func_description)
                evaluation_result = evaluate(id, inference_result, ground_truth, func_description)
                evaluation_result["id"] = id
                evaluation_results.append(evaluation_result)
                f_evaluation_out.seek(0)
                f_evaluation_out.truncate()
                for evaluation_result in evaluation_results:
                    f_evaluation_out.write(json.dumps(evaluation_result, ensure_ascii=False) + '\n')
                f_evaluation_out.flush()
            evaluation_results = sorted(evaluation_results, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
            f_evaluation_out.seek(0)
            f_evaluation_out.truncate()
            for evaluation_result in evaluation_results:
                f_evaluation_out.write(json.dumps(evaluation_result, ensure_ascii=False) + '\n')
            f_evaluation_out.flush()
    if requires_score:
        with open(score_path, 'w', encoding='utf-8') as f_score_out,\
                open(evaluation_result_path, 'r', encoding='utf-8') as f_evaluation_in:
            if not f_evaluation_in.readable():
                print(f"Error: evaluation result file {evaluation_result_path} is not readable.")
                continue
            total_cases = 0
            correct_cases = 0
            wrong_cases = []
            for line in f_evaluation_in:
                evaluation_entry = json.loads(line)
                total_cases += 1
                if evaluation_entry['valid']:
                    correct_cases += 1
                else:
                    wrong_cases.append(evaluation_entry)
            accuracy = correct_cases / total_cases if total_cases > 0 else 0.0
            score_result = {
                "accuracy": accuracy,
                "total_cases": total_cases,
                "correct_cases": correct_cases,
            }
            f_score_out.write(json.dumps(score_result, ensure_ascii=False, indent=2) + '\n')
            for wrong_case in wrong_cases:
                f_score_out.write(json.dumps(wrong_case, ensure_ascii=False) + '\n')
            f_score_out.flush()
            print(f"Score result written to {score_path}: {score_result}")
    print(f"Completed processing for config: {config}")




