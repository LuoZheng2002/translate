from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
        
def local_inference(generator, input_messages: list) -> str:
    os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"
    
    # Model name
    model_id = "ibm-granite/granite-3.1-8b-instruct"

    # Optional: specify where to download the model (important on HPCs)
    # model_dir = "/path/to/your/llm_storage"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",         # automatically shard across GPUs
        torch_dtype="auto",        # bfloat16 if supported
        offload_folder="/work/nvme/bfdz/zluo8/hf_offload",  # optional CPU offload
    )

    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Run inference
    prompt = "Write a short poem about the moon and the sea."
    output = generator(prompt, max_new_tokens=100, temperature=0.7)
    print(output[0]["generated_text"])