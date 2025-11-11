from parse_dataset import load_json_lines
import json
from config import *
from parse_ast import *
import re
from call_llm import make_chat_pipeline
from models.model_factory import create_model_interface
# def gen_developer_prompt(function_calls: list, prompt_passing_in_english: bool, model: LocalModel = None):
#     """
#     Generate system prompt for the model.

#     Args:
#         function_calls: List of available function definitions
#         prompt_passing_in_english: Whether to request English parameter passing
#         model: Optional LocalModel to customize prompt for specific models

#     Returns:
#         System prompt as a string
#     """
#     function_calls_json = json.dumps(function_calls, ensure_ascii=False, indent=2)
#     passing_in_english_prompt = " Pass in all parameters in function calls in English." if prompt_passing_in_english else ""

#     # Check if this is a Granite model
#     if model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
#         # Granite should output in JSON format (list of function call objects)
#         return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

# You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.{passing_in_english_prompt}

# At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\'s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

# Here is a list of functions in json format that you can invoke.
# {function_calls_json}
# '''
#     else:
#         # For API models, use Python function call syntax
#         return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

# You should only return the function calls in your response.

# If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the response.{passing_in_english_prompt}

# At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\"s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

# Here is a list of functions in json format that you can invoke.
# {function_calls_json}
# '''


def inference(model: Model, test_entry: dict, model_interface=None):
    """
    Run inference on a single test entry.

    Args:
        model: Model configuration (ApiModel or LocalModelStruct)
        test_entry: Test case entry with 'function' and 'question' fields
        model_interface: Optional pre-created model interface. If None, will be created.

    Returns:
        Dictionary with 'id' and 'result' (raw model output)
    """
    functions = test_entry['function']
    user_question = test_entry["question"][0][0]['content']

    # Create model interface if not provided
    if model_interface is None:
        model_interface = create_model_interface(model)

    # Run inference with new interface (accepts functions directly)
    result = model_interface.infer(
        functions=functions,
        user_query=user_question,
        prompt_passing_in_english=True
    )

    result_to_write = {
        "id": test_entry["id"],
        "result": result
    }
    return result_to_write



# Run inference

def create_chat_pipeline(config):
    # create chat pipeline for local model if needed
    match config.model:
        case ApiModel():
            pass
        case LocalModelStruct(model=local_model):
            # prepare the generator
            model_pipeline = make_chat_pipeline(local_model)
            config.model.generator = model_pipeline
            assert config.model.generator is not None, "Local model generator is not initialized."
        case _:
            raise ValueError(f"Unsupported model struct: {config.model}")


def cleanup_gpu_memory(config):
    """
    Explicitly free GPU memory for local models.
    Call this between configs to prevent memory accumulation.
    """
    import torch
    import gc

    if isinstance(config.model, LocalModelStruct) and config.model.generator is not None:
        # Delete the generator (which holds reference to the model)
        config.model.generator = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()
        print("GPU memory cleared.")


# Global variable to track if pipeline is initialized (reuse across configs)
_global_pipeline = None
_global_pipeline_model = None

def get_or_create_pipeline(local_model: LocalModel):
    """
    Get or create a pipeline for a local model.
    Reuses the same pipeline across configs with the same model.

    Guarantees: If you switch to a different model, the previous model's memory
    is immediately freed (assumes current model will never be used again in this run).
    """
    import torch
    import gc

    global _global_pipeline, _global_pipeline_model

    # If we have a pipeline for the same model, reuse it
    if _global_pipeline is not None and _global_pipeline_model == local_model:
        print(f"Reusing existing pipeline for {local_model.value}")
        return _global_pipeline

    # Different model detected - aggressive cleanup of old pipeline
    if _global_pipeline is not None:
        print(f"Switching from {_global_pipeline_model.value} to {local_model.value}")
        print(f"Freeing memory from previous model...")

        # Delete the generator and model references
        _global_pipeline = None
        _global_pipeline_model = None

        # Force immediate garbage collection
        gc.collect()
        gc.collect()  # Run twice to handle reference cycles

        # Clear CUDA cache - this is the key step
        torch.cuda.empty_cache()

        print(f"Memory freed. Loading new model...")

    # Create new pipeline for the new model
    print(f"Creating pipeline for {local_model.value}")
    _global_pipeline = make_chat_pipeline(local_model)
    _global_pipeline_model = local_model
    return _global_pipeline

for config in configs:
    print(f"Processing config: {config}")
    # config is composed of (model, translate_mode, add_noise_mode)

    # process model configuration
    # map model to model_postfix
    match config.model:
        case ApiModel() as api_model:
            match api_model:
                case ApiModel.GPT_4O_MINI:
                    model_postfix = "_gpt4o_mini"
                case ApiModel.CLAUDE_SONNET:
                    model_postfix = "_claude_sonnet"
                case ApiModel.CLAUDE_HAIKU:
                    model_postfix = "_claude_haiku"
                case _:
                    raise ValueError(f"Unsupported API model: {model}")
        case LocalModelStruct(model=model):
            match model:
                case LocalModel.GRANITE_3_1_8B_INSTRUCT:
                    model_postfix = "_granite"
                case _:
                    raise ValueError(f"Unsupported local model: {model}")
        case _:
            raise ValueError(f"Unsupported model struct: {config.model}")
    
        
    # map translate_info to language_postfix, translate_dataset_prefix, translate_mode_prefix
    match config.translate_mode:
        case Translated(language, option):
            match language:
                case Language.CHINESE:
                    language_postfix = "_zh"
                case Language.HINDI:
                    language_postfix = "_hi"
            match option:
                case TranslateOption.DATASET_FULLY_TRANSLATED:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_d" # default
                case TranslateOption.DATASET_FULLY_TRANSLATED_PROMPT_TRANSLATE:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_pt"  # prompt translate
                case TranslateOption.DATASET_PARTIALLY_TRANSLATED:
                    translate_dataset_postfix = "_partial"
                    translate_mode_postfix = "_par" # partial
        case NotTranslated():
            language_postfix = ""
            translate_dataset_postfix = ""
            translate_mode_postfix = ""
    match config.add_noise_mode:
        case AddNoiseMode.NO_NOISE:
            noise_postfix = ""
        case AddNoiseMode.SYNONYM:
            noise_postfix = "_syno"
        case AddNoiseMode.PARAPHRASE:
            noise_postfix = "_para"
    
    
    dataset_path = f"dataset/BFCL_v4_multiple{language_postfix}{translate_dataset_postfix}{noise_postfix}.json"
    ground_truth_path = f"dataset/possible_answer/BFCL_v4_multiple.json"
    inference_raw_result_path = f"result/inference_raw/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"
    inference_json_result_path = f"result/inference_json/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"
    evaluation_result_path = f"result/evaluation/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"
    score_path = f"result/score/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"
    with open(dataset_path, 'r', encoding='utf-8') as f_dataset:
        test_cases = load_json_lines(f_dataset)
    # test_cases = test_cases[:1]
    # open or create the inference result file
    if requires_inference_raw:
        chat_pipeline_created = False
        inference_raw_results = []
        existing_inference_ids = set()
        try:
            with open(inference_raw_result_path, 'r', encoding='utf-8') as f_inference_raw_out:
                if f_inference_raw_out.readable():
                    # read all lines and parse as list of dict
                    for line in f_inference_raw_out:
                        line_json = json.loads(line)
                        id = line_json["id"]
                        inference_raw_results.append(line_json)
                        existing_inference_ids.add(id)
        except FileNotFoundError:
            print(f"Inference result file {inference_raw_result_path} not found. It will be created.")

        # Batch processing configuration
        batch_size = 8  # Process 8 cases at a time for better GPU utilization
        printed_warning = False

        with open(inference_raw_result_path, 'w', encoding='utf-8') as f_inference_raw_out:
            # Filter cases that haven't been processed yet
            cases_to_process = [case for case in test_cases if case['id'] not in existing_inference_ids]

            if not printed_warning and len(cases_to_process) < len(test_cases):
                print(f"Warning: some test cases already exist in inference result file. Skipping {len(test_cases) - len(cases_to_process)} cases.")
                printed_warning = True

            # Determine if using API or local model
            is_api_model = isinstance(config.model, ApiModel)
            is_local_model = isinstance(config.model, LocalModelStruct)

            # Get or create chat pipeline for local models (reuses across configs)
            if is_local_model and len(cases_to_process) > 0:
                local_model = config.model.model
                generator = get_or_create_pipeline(local_model)
                config.model.generator = generator  # Assign to config for consistency
                chat_pipeline_created = True

            # Process in batches
            for batch_start in range(0, len(cases_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(cases_to_process))
                batch_cases = cases_to_process[batch_start:batch_end]

                print(f"\nProcessing batch {batch_start // batch_size + 1}: cases {batch_start} to {batch_end}")

                # Prepare batch data
                batch_functions_list = []
                batch_user_queries = []
                for case in batch_cases:
                    functions = case['function']
                    user_question = case["question"][0][0]['content']
                    batch_functions_list.append(functions)
                    batch_user_queries.append(user_question)

                # Create or reuse model interface
                if is_api_model:
                    model_interface = create_model_interface(config.model)
                    # For API models: process each case individually (handles concurrent requests internally)
                    print(f"Sending {len(batch_cases)} concurrent API requests...")
                    batch_results = []
                    for functions, user_query in zip(batch_functions_list, batch_user_queries):
                        result = model_interface.infer(
                            functions=functions,
                            user_query=user_query,
                            prompt_passing_in_english=True
                        )
                        batch_results.append(result)
                    print(f"Received {len(batch_results)} API responses.")

                elif is_local_model:
                    # For local models: create interface with generator
                    local_model = config.model.model
                    model_interface = create_model_interface(config.model, generator=config.model.generator)

                    if local_model == LocalModel.GRANITE_3_1_8B_INSTRUCT:
                        # Use batch processing for Granite (true batch inference)
                        batch_results = model_interface.infer_batch(
                            functions_list=batch_functions_list,
                            user_queries=batch_user_queries,
                            prompt_passing_in_english=True
                        )
                    else:
                        raise ValueError(f"Unsupported local model in batch processing: {local_model}")
                else:
                    raise ValueError(f"Unsupported model type: {type(config.model)}")

                # Process results
                for case, result in zip(batch_cases, batch_results):
                    print(f"Inferencing case id {case['id']}, question: {case['question'][0][0]['content']}")
                    print("Answer: ", result)

                    result_to_write = {
                        "id": case["id"],
                        "result": result
                    }
                    inference_raw_results.append(result_to_write)

                # Write batch results to file
                f_inference_raw_out.seek(0)
                f_inference_raw_out.truncate()
                for result in inference_raw_results:
                    f_inference_raw_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_inference_raw_out.flush()

            # Final sort and write
            if len(inference_raw_results) > 0:
                inference_raw_results = sorted(inference_raw_results, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
                f_inference_raw_out.seek(0)
                f_inference_raw_out.truncate()
                for result in inference_raw_results:
                    f_inference_raw_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_inference_raw_out.flush()
    if requires_inference_json:
        inference_json_results = []
        existing_inference_json_ids = set()
        printed_warning = False
        with open(inference_json_result_path, 'w', encoding='utf-8') as f_inference_json_out:
            for inference_raw in inference_raw_results:
                if inference_raw['id'] in existing_inference_json_ids:
                    if not printed_warning:
                        print(f"Warning: some test cases already exist in inference json result file. Skipping.")
                        printed_warning = True
                    continue
                # convert raw result to json format
                #                
                id = inference_raw['id']
                decoded_output = raw_to_json(config.model, id, inference_raw['result'])
                inference_json_entry = {
                    "id": id,
                    "result": decoded_output
                }
                inference_json_results.append(inference_json_entry)
                f_inference_json_out.write(json.dumps(inference_json_entry, ensure_ascii=False) + '\n')
                f_inference_json_out.flush()
            inference_json_results = sorted(inference_json_results, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
            f_inference_json_out.seek(0)
            f_inference_json_out.truncate()
            for result in inference_json_results:
                f_inference_json_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_inference_json_out.flush()
    if requires_evaluation:
        evaluation_results = []
        existing_evaluation_ids = set()
        # try:
        #     with open(evaluation_result_path, 'r', encoding='utf-8') as f_evaluation_out:
        #         if f_evaluation_out.readable():
        #             # read all lines and parse as list of dict
        #             for line in f_evaluation_out:
        #                 line_json = json.loads(line)
        #                 id = line_json["id"]                        
        #                 evaluation_results.append(line_json)
        #                 existing_evaluation_ids.add(id)
        # except FileNotFoundError:
        #     print(f"Evaluation result file {evaluation_result_path} not found. It will be created.")
        printed_warning = False

        # prepare ground truth
        ground_truths = []
        with open(ground_truth_path, 'r', encoding='utf-8') as f_ground_truth_in:
            for line in f_ground_truth_in:
                ground_truths.append(json.loads(line))
        with open(evaluation_result_path, 'w', encoding='utf-8') as f_evaluation_out:
            for (inference_json_line, ground_truth_line, test_case) in zip(inference_json_results, ground_truths, test_cases):
                id = inference_json_line["id"]
                if id in existing_evaluation_ids:
                    if not printed_warning:
                        print(f"Warning: some test cases already exist in evaluation result file. Skipping.")
                        printed_warning = True
                    continue
                assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
                assert id == test_case["id"], f"Mismatch in IDs: {id} vs {test_case['id']}"
                inference_json_result = inference_json_line["result"]
                ground_truth = ground_truth_line["ground_truth"]
                func_description = test_case['function']
                # print("func_description: ", func_description)
                
                evaluation_result = evaluate_json(id, inference_json_result, ground_truth, func_description)
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




