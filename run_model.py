from vllm import LLM, SamplingParams

import os
os.environ["HF_HUB_CACHE"] = "/work/nvme/bfdz/zluo8/huggingface"


llm = LLM("Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")

params = SamplingParams(temperature=0.7, max_tokens=128)

prompt = "Explain quantum entanglement in simple terms."
outputs = llm.generate([prompt], params)

print(outputs[0].outputs[0].text)
