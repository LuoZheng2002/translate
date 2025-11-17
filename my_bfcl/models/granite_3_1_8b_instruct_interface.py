"""
Interface for IBM Granite 3.1 8B Instruct local model.

Handles:
- Local model inference via generator pipeline
- Granite chat template formatting
- Output parsing from JSON format
"""

import json
from typing import List, Dict, Any, Union
from models.base import ModelInterface

try:
    from config import LocalModel
except ImportError:
    LocalModel = None


class Granite3_1_8BInstructInterface(ModelInterface):
    """Handler for IBM Granite 3.1 8B Instruct local model."""

    def __init__(self, generator):
        """
        Initialize the Granite interface.

        Args:
            generator: Optional pre-initialized generator from make_chat_pipeline()
                      If None, you must provide it via set_generator() before calling infer()
        """
        self.generator = generator
        self.model_id = "ibm-granite/granite-3.1-8b-instruct"

    # def set_generator(self, generator):
    #     """
    #     Set the model generator pipeline.

    #     Args:
    #         generator: Generator from make_chat_pipeline()
    #     """
    #     self.generator = generator

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None) -> str:
        """
        Run inference with Granite model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: Should be LocalModel.GRANITE_3_1_8B_INSTRUCT (for system prompt generation)

        Returns:
            Raw model output as a string
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Call set_generator() first.")

        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Format the input using Granite chat template with functions
        template = self._format_granite_chat_template(
            system_prompt=system_prompt,
            user_query=user_query,
            functions=functions,
            add_generation_prompt=True
        )

        # Send template to generator and get response
        result = self.generator.send(template)
        return result

    def infer_batch(self, functions_list: List[List[Dict[str, Any]]],
                   user_queries: List[str],
                   prompt_passing_in_english: bool = True) -> List[str]:
        """
        Run batch inference with Granite model.

        Args:
            functions_list: List of function lists (one per query)
            user_queries: List of user queries as strings
            prompt_passing_in_english: Whether to request English parameter passing

        Returns:
            List of raw model outputs as strings
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Call set_generator() first.")

        if len(functions_list) != len(user_queries):
            raise ValueError("functions_list and user_queries must have same length")

        # Format all templates
        templates = []
        for functions, user_query in zip(functions_list, user_queries):
            system_prompt = self._generate_system_prompt(
                functions=functions,
                prompt_passing_in_english=prompt_passing_in_english
            )
            template = self._format_granite_chat_template(
                system_prompt=system_prompt,
                user_query=user_query,
                functions=functions,
                add_generation_prompt=True
            )
            templates.append(template)

        # Send batch to generator and get responses
        results = self.generator.send(templates)
        return results

    def infer_with_functions(self, system_prompt: str, user_query: str,
                           functions: List[Dict[str, Any]]) -> str:
        """
        Run inference with explicit system prompt and function definitions (tool calling).

        Args:
            system_prompt: System prompt as a string
            user_query: User query as a string
            functions: List of function definitions in JSON format

        Returns:
            Raw model output as a string
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Call set_generator() first.")

        # Format with function definitions
        template = self._format_granite_chat_template(
            system_prompt=system_prompt,
            user_query=user_query,
            functions=functions,
            add_generation_prompt=True
        )

        # Send template to generator and get response
        result = self.generator.send(template)
        return result

    def infer_batch_with_functions(self, system_prompts: List[str],
                                   user_queries: List[str],
                                   batch_functions: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Run batch inference with explicit system prompts and function definitions.

        Args:
            system_prompts: List of system prompts as strings
            user_queries: List of user queries as strings
            batch_functions: List of function definition lists

        Returns:
            List of raw model outputs as strings
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized. Call set_generator() first.")

        if not (len(system_prompts) == len(user_queries) == len(batch_functions)):
            raise ValueError("All input lists must have same length")

        # Format all templates with their respective functions
        templates = []
        for system_prompt, user_query, functions in zip(system_prompts, user_queries, batch_functions):
            template = self._format_granite_chat_template(
                system_prompt=system_prompt,
                user_query=user_query,
                functions=functions,
                add_generation_prompt=True
            )
            templates.append(template)

        # Send batch to generator and get responses
        results = self.generator.send(templates)
        return results

    def parse_output(self, raw_output: str) -> Union[List[Dict[str, Dict[str, Any]]], str]:
        """
        Parse raw output from Granite model using parse_ast.py strategy.

        Granite outputs in JSON format:
        <tool_call>[{"name": "function_name", "arguments": {"param1": value1, ...}}, ...]

        Follows the same parsing strategy as parse_ast.py's raw_to_json() for Granite models.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format: [{func_name: {arguments}}, ...]
            Returns error string if parsing fails (matching raw_to_json behavior)
        """
        # Parse Granite model's output format: <tool_call>[{...}] (from parse_ast.py:135)
        model_result_raw = raw_output.strip()

        # Remove <tool_call> wrapper if present (from parse_ast.py:138-139)
        if model_result_raw.startswith("<tool_call>"):
            model_result_raw = model_result_raw[len("<tool_call>"):]

        # Strip backticks and whitespace (from parse_ast.py:141)
        model_result_raw = model_result_raw.strip("`\n ")

        # Add brackets if missing (from parse_ast.py:144-147)
        if not model_result_raw.startswith("["):
            model_result_raw = "[" + model_result_raw
        if not model_result_raw.endswith("]"):
            model_result_raw = model_result_raw + "]"

        try:
            # Parse the JSON array (from parse_ast.py:151)
            tool_calls = json.loads(model_result_raw)
        except json.JSONDecodeError:
            return f"Failed to decode JSON: Invalid JSON format."

        # Convert Granite format to desired format (from parse_ast.py:156-166)
        extracted = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    func_name = tool_call["name"]
                    func_args = tool_call["arguments"]
                    extracted.append({func_name: func_args})
                else:
                    return f"Failed to decode JSON: Invalid tool call structure."
        else:
            return f"Failed to decode JSON: Expected a list of tool calls."

        return extracted

    def _generate_system_prompt(self, functions: List[Dict[str, Any]],
                               prompt_passing_in_english: bool = True) -> str:
        """
        Generate system prompt for Granite model based on available functions.

        Adapted from main.py's gen_developer_prompt() function.

        Args:
            functions: List of available function definitions
            prompt_passing_in_english: Whether to request English parameter passing

        Returns:
            System prompt as a string
        """
        function_calls_json = json.dumps(functions, ensure_ascii=False, indent=2)
        passing_in_english_prompt = (
            " IMPORTANT: Pass in all parameters in function calls in English."
            if prompt_passing_in_english
            else ""
        )

        # Granite format - JSON output
        return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response, in JSON format as a list where each element has the format {{"name": "function_name", "arguments": {{param1: value1, param2: value2, ...}}}}.{passing_in_english_prompt}

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user\'s request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''

    def _format_granite_chat_template(self, system_prompt: str, user_query: str,
                                     functions: List[Dict[str, Any]] = None,
                                     add_generation_prompt: bool = True) -> str:
        """
        Format messages using the Granite chat template.

        Args:
            system_prompt: System prompt as a string
            user_query: User query as a string
            functions: Optional list of function definitions for tool calling
            add_generation_prompt: Whether to add the generation prompt at the end

        Returns:
            Formatted prompt string using Granite's chat template
        """
        formatted_prompt = ""

        # Use provided system prompt
        system_content = system_prompt if system_prompt else (
            "Knowledge Cutoff Date: April 2024.\n"
            "Today's Date: April 29, 2025.\n"
            "You are Granite, developed by IBM."
        )

        # Add system prompt with Granite tags
        formatted_prompt += (
            f"<|start_of_role|>system<|end_of_role|>{system_content}<|end_of_text|>\n"
        )

        # Add tools section if functions are provided
        if functions:
            formatted_prompt += (
                "<|start_of_role|>tools<|end_of_role|>"
                + json.dumps(functions, indent=4)
                + "<|end_of_text|>\n"
            )

        # Add user message
        formatted_prompt += (
            "<|start_of_role|>user<|end_of_role|>"
            + user_query
            + "<|end_of_text|>\n"
        )

        # Add generation prompt if requested
        if add_generation_prompt:
            formatted_prompt += "<|start_of_role|>assistant<|end_of_role|>"

        return formatted_prompt
