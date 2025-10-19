import os
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model name
model_id = "Qwen/Qwen2.5-72B"

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
output = generator(prompt, max_new_tokens=10, temperature=0.7)
print(output[0]["generated_text"])
