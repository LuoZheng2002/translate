
import json

import re
from call_llm import gpt_4o_mini_inference
from parse_dataset import load_json_lines


system_prompt = '''
You are a helpful assistant helping rephrasing user requests, while accurately preserving their meaning, including numbers and names if exist. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations. Consider the example below:

USER: Can I find the dimensions and properties of a triangle, if it is known that its three sides are 5 units, 4 units and 3 units long?

ASSISTANT: What are the dimensions and
properties of a triangle whose three sides
are 5, 4 and 3 units long?
'''

def generate_noisy_case(question: str) -> str:
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    noisy_question = gpt_4o_mini_inference(input_messages)
    return noisy_question

postfix_to_generate = [
    ""
]

for postfix in postfix_to_generate:
    print(f"Generating noisy dataset for postfix: {postfix}")
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    noisy_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_noisy.json'
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)
    noisy_data = []
    existing_indices = []
    try:
        with open(noisy_dataset_path, 'r', encoding='utf-8') as f:
            noisy_data = load_json_lines(f)
            existing_indices = [item['id'] for item in noisy_data]
    except FileNotFoundError:
        print(f"No existing noisy dataset found at {noisy_dataset_path}. A new one will be created.")
    with open(noisy_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        for item in original_data:
            id = item['id']
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {noisy_dataset_path}.")
                    warning_printed = True
                continue
            noisy_question = generate_noisy_case(item['question'][0][0]['content'])
            noisy_item = item.copy()
            noisy_item['question'][0][0]['content'] = noisy_question
            noisy_data.append(noisy_item)
            f.seek(0)
            f.truncate()
            for n in noisy_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()
        # sort
        noisy_data = sorted(noisy_data, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
        f.seek(0)
        f.truncate()
        for n in noisy_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    print(f"Noisy dataset with {len(noisy_data)} items saved to {noisy_dataset_path}.")
    