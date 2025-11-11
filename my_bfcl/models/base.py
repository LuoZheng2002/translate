"""
Abstract base classes defining the interface for all model handlers.

Each model-specific interface should inherit from ModelInterface and implement:
1. infer(): Takes system prompt and user query, returns raw model output
2. parse_output(): Takes raw output, returns parsed function calls
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ModelInterface(ABC):
    """
    Abstract base class for all model handlers.

    Each model implementation should:
    1. Accept system_prompt and user_query as strings
    2. Return raw model output as string from infer()
    3. Parse raw output to standardized function call list format
    """

    @abstractmethod
    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None) -> str:
        """
        Run inference with the model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query/question as a string
            prompt_passing_in_english: Whether to request English parameter passing (default: True)
            model: Optional model type for customizing system prompt (LocalModel enum for Granite)

        Returns:
            Raw model output as a string
        """
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> List[Dict[str, Dict[str, Any]]]:
        """
        Parse raw model output to standardized function call format.

        This method follows the parsing strategy from parse_ast.py's raw_to_json() function.

        Args:
            raw_output: Raw string output from the model

        Returns:
            List of function call dictionaries in format:
            [
                {
                    "function_name": {
                        "param1": value1,
                        "param2": value2,
                        ...
                    }
                },
                ...
            ]

            For error cases, returns a string describing the error (same as raw_to_json()).

        Raises:
            ValueError: If output cannot be parsed (alternative to returning error string)
        """
        pass
