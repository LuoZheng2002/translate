
from config import ApiModel, Model
from dotenv import load_dotenv
import os


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
                messages=input_messages
            )

            return response.choices[0].message.content

        case ApiModel.CLAUDE_SONNET | ApiModel.CLAUDE_HAIKU:
            # Use Anthropic client
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

            client = Anthropic(api_key=api_key)

            # Extract system message (first "system" if any)
            system_message = None
            messages_for_claude = []
            for m in input_messages:
                if m["role"] == "system" and system_message is None:
                    system_message = m["content"]
                elif m["role"] in {"user", "assistant"}:
                    messages_for_claude.append(
                        {"role": m["role"], "content": m["content"]}
                    )

            if not any(m["role"] == "user" for m in messages_for_claude):
                raise ValueError("Claude API requires at least one 'user' message")

            response = client.messages.create(
                model=model.value,
                max_tokens=1024,
                system=system_message,
                messages=messages_for_claude
            )

            # Anthropic returns structured content (list of message blocks)
            return response.content[0].text if response.content else ""

        case _:
            raise ValueError(f"Unsupported model: {model}")
        

def make_chat_pipeline(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    """
    Returns a generator function that takes (system, user) input pairs and yields model responses.
    """
    print(f"Loading local model: {model_id}")
    # --- Environment setup ---
    os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # --- (Optional) quantization configuration ---
    # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        offload_folder="/work/nvme/bfdz/zluo8/hf_offload",
        # quantization_config=bnb_config,
    )

    model.eval()

    # --- Define the generator pipeline ---
    def chat_generator():
        pair = yield  # Initial yield to start the generator
        while True:
            # Wait for a pair of system and user messages
            if pair is None:
                pair = yield
                continue
            system_prompt, user_prompt = pair

            # Construct message list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # print("system prompt:\n", system_prompt)
            # print("user prompt:\n", user_prompt)

            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Generate
            print("Generating response...")
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
            print("Generation complete.")
            # Decode
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # print("response:\n", response)

            # Yield response
            pair = yield response

    # Initialize and prime the generator
    gen = chat_generator()
    next(gen)
    print("Local model loaded and generator is ready.")
    return gen
