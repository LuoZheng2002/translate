import ast
def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        # output = tuple(resolve_ast_by_type(v) for v in value.elts) 
        output = [resolve_ast_by_type(v) for v in value.elts] # Zheng: Changed to list to match ground truth
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}



    

   


def evaluate(
    case_id,
    model_result,
    possible_answer,
    func_description,
):
    """Helper method to process a single AST entry."""
    model_result_item_raw = model_result    
    model_result = model_result.strip("`\n ")
    if not model_result.startswith("["):
        model_result = "[" + model_result
    if not model_result.endswith("]"):
        model_result = model_result + "]"
    # We only want to remove wrapping quotes that could have been added by the model.
    cleaned_input = model_result.strip().strip("'")
    try:
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                if not isinstance(elem, ast.Call):
                    raise Exception(f"Expected AST Call node, but got {type(elem)}")
                extracted.append(resolve_ast_call(elem))
        decoded_output = extracted
        
    except Exception as e:
        return {
            "id": case_id,
            "valid": False,
            "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
            "error_type": "ast_decoder:decoder_failed",
            "model_result_raw": model_result_item_raw,
            "possible_answer": possible_answer,
        }
    
    model_result = decoded_output
    if len(model_result) != 1:
        return {
            "id": case_id,
            "valid": False,
            "error": [f"Expected exactly one AST entry, but got {len(model_result)}."],
            "error_type": "ast_checker:invalid_entry_count",
            "model_result_raw": model_result_item_raw,
            "model_result_decoded": model_result,
            "possible_answer": possible_answer,
        }
    model_result = model_result[0]
    possible_answer = possible_answer[0]

    # print("model_result:", model_result)
    # print("possible_answer:", possible_answer)
    assert(len(possible_answer) == 1)
    assert(len(model_result) == 1)
    # print("model_result:", model_result)
    
    # print("possible_answer:", possible_answer)
    # print("possible_answer keys:", possible_answer.keys()   )
    # Extract function name and parameters details
    possible_answer_func_name = next(iter(possible_answer))
    model_result_func_name = next(iter(model_result))
    # print("func_name:", possible_answer_func_name)
    if possible_answer_func_name != model_result_func_name:
        return {
            "id": case_id,
            "valid": False,
            "error": [f"Function name mismatch. Expected {repr(possible_answer_func_name)}, but got {repr(model_result_func_name)}."],
            "error_type": "simple_function_checker:wrong_func_name",
            "model_result_raw": model_result_item_raw,
            "model_result_decoded": model_result,
            "possible_answer": possible_answer,
        }
    # first see if function name matches

    # param_details = func_description["parameters"]["properties"]
    for func in func_description:
        if func['name'] == possible_answer_func_name:
            # print("func parameters:", func['parameters'].keys())
            required_params = func['parameters']['required']
            break

    # Check for required parameters in model output
    model_params = model_result[possible_answer_func_name]
    for param in required_params:
        if param not in model_params:
            return {
                "id": case_id,
                "valid": False,
                "error": [f"Missing required parameter: {repr(param)}."],
                "error_type": "simple_function_checker:missing_required",
                "model_result_raw": model_result_item_raw,
                "model_result_decoded": model_result,
                "possible_answer": possible_answer,
            }
        
    possible_answer_params = possible_answer[possible_answer_func_name]

    # Validate types and values for each parameter in model output
    for param, value in model_params.items():
        if param not in possible_answer_params:
            print("possible_answer keys:", possible_answer.keys())
            print("param: ", param)
            return {
                "id": case_id,
                "valid": False,
                "error": [f"Unexpected parameter: {repr(param)}."],
                "error_type": "simple_function_checker:unexpected_param",
                "model_result_raw": model_result_item_raw,
                "model_result_decoded": model_result,
                "possible_answer": possible_answer,
            }
        # Check if the value is within the possible answers
        if value not in possible_answer_params[param]:
            return {
                "id": case_id,
                "valid": False,
                "error": [f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer_params[param]}."],
                "error_type": "value_error:others",
                "model_result_raw": model_result_item_raw,
                "model_result_decoded": model_result,
                "possible_answer": possible_answer,
            }
    return {
        "id": case_id,
        "valid": True,
        "model_result_raw": model_result_item_raw,
        "model_result_decoded": model_result,
        "possible_answer": possible_answer,
    }