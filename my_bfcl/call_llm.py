
from config import Model
from dotenv import load_dotenv
import os


def api_inference(model: Model, input_messages: list) -> str:
    load_dotenv(dotenv_path=".env")
    match model:
        case Model.GPT_4O_MINI:
            # Use OpenAI client
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model.value,
                messages=input_messages
            )
            return response.choices[0].message.content

        case Model.CLAUDE_SONNET | Model.CLAUDE_HAIKU:
            # Use Anthropic client
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key)

            # Convert OpenAI-style messages to a single prompt (Claude expects text)
            # prompt = "\n".join(
            #     f"{m['role'].capitalize()}: {m['content']}" for m in input_messages
            # )
            system_message = input_messages[0]['content']
            user_message = input_messages[1]['content']

            response = client.messages.create(
                model=model.value,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                system=system_message
            )
            return response.content[0].text
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

    # --- Define the generator pipeline ---
    def chat_generator():
        while True:
            # Wait for a pair of system and user messages
            pair = yield  # Receive input
            if pair is None:
                continue
            system_prompt, user_prompt = pair

            # Construct message list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Generate
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Yield response
            yield response

    # Initialize and prime the generator
    gen = chat_generator()
    next(gen)
    print("Local model loaded and generator is ready.")
    return gen
