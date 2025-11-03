from parse_dataset import load_json_lines
import json
from config import *
from parse_ast import *
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

def gpt_4o_mini_inference(input_messages: list) -> str:
    from openai import OpenAI
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path=".env")  # reads .env into environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialize client (API key can be set as env variable or directly)
    client = OpenAI(api_key=api_key)  # or omit api_key if you set it in the environment

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=input_messages
    )
    processed_line = response.choices[0].message.content
    return processed_line


# Run inference

for config in configs:
    match config.translate_info:
        case Translated(language=lang, translate_mode=mode):
            match lang:
                case Language.CHINESE:
                    language_postfix = "_zh"
                case Language.HINDI:
                    language_postfix = "_hi"
            match mode:
                case TranslateMode.DATASET_FULLY_TRANSLATED_PROMPT_DEFAULT:
                    translate_postfix = "_full"
                case TranslateMode.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE:
                    translate_postfix = "_full"
                case TranslateMode.DATASET_PARTIALLY_TRANSLATED:
                    translate_postfix = "_partial"
        case NotTranslated():
            language_postfix = ""
            translate_postfix = ""
    match config.add_noise_mode:
        case AddNoiseMode.NO_NOISE:
            noise_postfix = ""
        case AddNoiseMode.ADD_NOISE:
            noise_postfix = "_noisy"
    dataset_path = f"dataset/BFCL_v4_multiple{language_postfix}{translate_postfix}{noise_postfix}.json"
    ground_truth_path = f"dataset/possible_answer/BFCL_v4_multiple.json"
    inference_result_path = f"result/inference/BFCL_v4_multiple{language_postfix}{translate_postfix}{noise_postfix}.json"
    score_result_path = f"result/score/BFCL_v4_multiple{language_postfix}{translate_postfix}{noise_postfix}.json"

    test_cases = load_json_lines(dataset_path)
    test_cases = test_cases[:1]
    # open or create the inference result file
    if requires_inference:
        results = []
        with open(inference_result_path, 'r', encoding='utf-8') as f_inference_out:
            if f_inference_out.readable():
                # read all lines and parse as list of dict
                for line in f_inference_out:
                    results.append(json.loads(line))
            existing_ids = {entry['id'] for entry in results}
        with open(inference_result_path, 'w', encoding='utf-8') as f_inference_out:
            for i, case in enumerate(test_cases):
                if case['id'] in existing_ids:
                    print(f"Warning: case id {case['id']} already exists in inference result file. Skipping.")
                    continue
                # the actual inference
                result = inference(case)
                results.append(result)
            # rewrite all results to the file
            results = sorted(results, key=lambda x: x["id"])
            f_inference_out.seek(0)
            for result in results:
                f_inference_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_inference_out.truncate()
    if requires_evaluation:
        with open(inference_result_path, 'r', encoding='utf-8') as f_inference_in, \
                open(score_result_path, 'w', encoding='utf-8') as f_score_out, \
                open(ground_truth_path, 'r', encoding='utf-8') as f_ground_truth_in:
            if not f_inference_in.readable():
                print(f"Error: inference result file {inference_result_path} is not readable.")
                continue
            inference_results = []
            ground_truths = []
            scores = []
            for line in f_inference_in:
                inference_results.append(json.loads(line))
            for line in f_ground_truth_in:
                ground_truths.append(json.loads(line))
            for (inference_line, ground_truth_line) in zip(inference_results, ground_truths):
                id = inference_line["id"]
                assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
                inference_result = inference_line["result"]
                ground_truth = ground_truth_line["ground_truth"]
                score_result = evaluate(inference_result, ground_truth)
                score_result["id"] = id
                scores.append(score_result)
            scores = sorted(scores, key=lambda x: x["id"])
            for score in scores:
                f_score_out.write(json.dumps(score, ensure_ascii=False) + '\n')
    print(f"Completed processing for config: {config}")
    exit(1)  # Zheng




