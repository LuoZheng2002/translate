from parse_dataset import load_json_lines
import json

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

def gen_prompt_dict(developer_prompt: str, user_question: str):
    developer_message = {"role": "developer", "content": developer_prompt}
    user_message = {"role": "user", "content": user_question}
    return {"message": [developer_message, user_message]}

def inference(test_entry: dict):
    print(test_entry)
    functions = test_entry["question"][0][0]['function']
    developer_prompt = gen_developer_prompt(
        function_calls=functions,
        prompt_passing_in_english=True
    )
    user_question = test_entry["question"][0][0]['content']
    prompt_dict = gen_prompt_dict(
        developer_prompt=developer_prompt,
        user_question=user_question
    )
    print("Prompt dict:")
    print(prompt_dict)
    # to do: call the LLM API with prompt_dict
    result = ""

    result_to_write = {
        "id": test_entry["id"],
        "result": result
    }
    return result_to_write


test_cases = load_json_lines("dataset/BFCL_v4_multiple.json")
for i, case in enumerate(test_cases):
    result = inference(case)
    print(f"Case {i} result: {result}")
    exit(1)  # Zheng Luo
