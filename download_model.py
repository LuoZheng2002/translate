from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-7B-Instruct"  # example model, pick your own

# Where to store model files
cache_dir = "/work/nvme/bfdz/zluo8/huggingface"

# Download model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

print("✅ Model and tokenizer downloaded to:", cache_dir)
