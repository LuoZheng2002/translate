import os
os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-72B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    offload_folder="/work/nvme/bfdz/zluo8/hf_offload",
    # quantization_config=bnb_config,
)

prompt = "Write a short poem about the moon and the sea."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
