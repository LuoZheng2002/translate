"""
Factory for creating model interface instances.

Provides a unified way to instantiate the correct model handler based on model type.
"""

from typing import Union, Optional, List, Dict, Any
from config import ApiModel, LocalModel, LocalModelStruct
from models.base import ModelInterface
from models.gpt_4o_mini_interface import GPT4oMiniInterface
from models.claude_sonnet_interface import ClaudeSonnetInterface
from models.claude_haiku_interface import ClaudeHaikuInterface
from models.deepseek_chat_interface import DeepseekChatInterface
from models.granite_3_1_8b_instruct_interface import Granite3_1_8BInstructInterface


def create_model_interface(model: Union[ApiModel, LocalModelStruct],
                          generator=None) -> ModelInterface:
    """
    Factory function to create appropriate model interface instance.

    Args:
        model: Either an ApiModel enum or LocalModelStruct instance
        generator: Optional pre-initialized generator for local models

    Returns:
        ModelInterface instance for the specified model

    Raises:
        ValueError: If model type is not supported
        EnvironmentError: If required API keys are missing
    """
    if isinstance(model, ApiModel):
        return _create_api_model_interface(model)
    elif isinstance(model, LocalModelStruct):
        return _create_local_model_interface(model.model, generator)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def _create_api_model_interface(model: ApiModel) -> ModelInterface:
    """
    Create interface for API-based models.

    Args:
        model: ApiModel enum value

    Returns:
        ModelInterface instance

    Raises:
        ValueError: If model is not supported
        EnvironmentError: If required API keys are missing
    """
    match model:
        case ApiModel.GPT_4O_MINI:
            return GPT4oMiniInterface()
        case ApiModel.CLAUDE_SONNET:
            return ClaudeSonnetInterface()
        case ApiModel.CLAUDE_HAIKU:
            return ClaudeHaikuInterface()
        case ApiModel.DEEPSEEK_CHAT:
            return DeepseekChatInterface()
        case _:
            raise ValueError(f"Unsupported API model: {model}")


def _create_local_model_interface(model: LocalModel, generator=None) -> ModelInterface:
    """
    Create interface for local models.

    Args:
        model: LocalModel enum value
        generator: Optional pre-initialized generator

    Returns:
        ModelInterface instance

    Raises:
        ValueError: If model is not supported
    """
    match model:
        case LocalModel.GRANITE_3_1_8B_INSTRUCT:
            return Granite3_1_8BInstructInterface(generator=generator)
        case _:
            raise ValueError(f"Unsupported local model: {model}")
