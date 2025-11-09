
from config import ApiModel, LocalModel, Model
from dotenv import load_dotenv
import os
import json


def api_inference(model: ApiModel, input_messages: list[dict]) -> str:
    """
    Run inference with either OpenAI or Anthropic API.

    Args:
        model: ApiModel enum value.
        input_messages: list of dicts like OpenAI's `messages` format.
            Example:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
    """
    load_dotenv(dotenv_path=".env")

    # --- Validate arguments ---
    if not isinstance(model, ApiModel):
        raise TypeError("`model` must be an ApiModel")

    if not isinstance(input_messages, list) or not all(isinstance(m, dict) for m in input_messages):
        raise TypeError("`input_messages` must be a list of dicts")

    valid_roles = {"system", "user", "assistant"}
    for i, m in enumerate(input_messages):
        if "role" not in m or "content" not in m:
            raise ValueError(f"Message at index {i} missing 'role' or 'content'")
        if m["role"] not in valid_roles:
            raise ValueError(f"Invalid role '{m['role']}' at index {i}")

    # --- Dispatch based on model ---
    match model:
        case ApiModel.GPT_4O_MINI:
            # Use OpenAI client
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not found in .env")

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model.value,
                messages=input_messages,
                temperature=0.001
            )

            return response.choices[0].message.content

        case ApiModel.CLAUDE_SONNET | ApiModel.CLAUDE_HAIKU:
            # Use Anthropic client
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

            client = Anthropic(api_key=api_key)

            # Extract system message (first "system" if any) and format it as Claude expects
            system_message = None
            messages_for_claude = []
            for m in input_messages:
                if m["role"] == "system" and system_message is None:
                    # Claude expects system message as a list of content blocks
                    system_message = [{"type": "text", "text": m["content"]}]
                elif m["role"] in {"user", "assistant"}:
                    # Claude expects message content as a list of content blocks
                    messages_for_claude.append(
                        {
                            "role": m["role"],
                            "content": [{"type": "text", "text": m["content"]}]
                        }
                    )

            if not any(m["role"] == "user" for m in messages_for_claude):
                raise ValueError("Claude API requires at least one 'user' message")

            # Build the API request
            kwargs = {
                "model": model.value,
                "max_tokens": 1024,
                "temperature": 0.001,
                "messages": messages_for_claude
            }

            # Include system message if it exists
            if system_message is not None:
                kwargs["system"] = system_message

            response = client.messages.create(**kwargs)

            # Anthropic returns structured content (list of message blocks)
            return response.content[0].text if response.content else ""

        case _:
            raise ValueError(f"Unsupported model: {model}")
        

def format_granite_chat_template(messages: list[dict], functions: list[dict] = None, add_generation_prompt: bool = True) -> str:
    """
    Format messages using the Granite chat template.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        functions: Optional list of function definitions for tool calling
        add_generation_prompt: Whether to add the generation prompt at the end

    Returns:
        Formatted prompt string using Granite's chat template
    """
    formatted_prompt = ""

    # Extract system message if present
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        messages_to_process = messages[1:]
    else:
        # Default system prompt for Granite
        system_prompt = (
            "Knowledge Cutoff Date: April 2024.\n"
            "Today's Date: April 29, 2025.\n"
            "You are Granite, developed by IBM."
        )
        if functions:
            system_prompt += (
                " You are a helpful AI assistant with access "
                "to the following tools. When a tool is required to answer the user's query, respond "
                "with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist "
                "in the provided list of tools, notify the user that you do not have the ability to fulfill the request."
            )
        messages_to_process = messages

    # Add the system message
    formatted_prompt += (
        f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
    )

    # Add tools section if functions are provided
    if functions:
        formatted_prompt += (
            "<|start_of_role|>tools<|end_of_role|>"
            + json.dumps(functions, indent=4)
            + "<|end_of_text|>\n"
        )

    # Add all messages
    for msg in messages_to_process:
        formatted_prompt += (
            "<|start_of_role|>"
            + msg["role"]
            + "<|end_of_role|>"
            + msg["content"]
            + "<|end_of_text|>\n"
        )

    # Add generation prompt if requested
    if add_generation_prompt:
        formatted_prompt += "<|start_of_role|>assistant<|end_of_role|>"

    return formatted_prompt


def make_chat_pipeline(model: LocalModel):
    """
    Returns a generator function that takes a populated template string and yields model responses.

    Args:
        model: A LocalModel enum value specifying which local model to use

    Returns:
        A generator that accepts a populated template string and yields responses
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Extract model_id from the enum
    model_id = model.value
    print(f"Loading local model: {model_id}")

    # --- Environment setup ---
    os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # --- (Optional) quantization configuration ---
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    # --- Load model ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        offload_folder="/work/nvme/bfdz/zluo8/hf_offload",
        # quantization_config=bnb_config,
    )

    hf_model.eval()

    # --- Define the generator pipeline ---
    def chat_generator():
        template = yield  # Initial yield to start the generator
        while True:
            # Wait for a populated template string
            if template is None:
                template = yield
                continue

            # Tokenize the populated template
            inputs = tokenizer(template, return_tensors="pt").to(hf_model.device)

            # Generate
            print("Generating response...")
            with torch.inference_mode():
                outputs = hf_model.generate(**inputs, max_new_tokens=100, temperature=0.001)
            print("Generation complete.")
            # Decode
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Yield response and wait for next template
            template = yield response

    # Initialize and prime the generator
    gen = chat_generator()
    next(gen)
    print("Local model loaded and generator is ready.")
    return gen
