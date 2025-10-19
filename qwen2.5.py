import os
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-72B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Optionally enable quantization (commented out here)
# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    offload_folder="/work/nvme/bfdz/zluo8/hf_offload",
    # quantization_config=bnb_config,
)

system_prompt = f"You are a helpful assistant that calls the following available functions."

# ---- Use chat template ----
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short poem about the moon and the sea."},
]

# Apply chat template to get the properly formatted input
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize the formatted input
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Generate the response
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
