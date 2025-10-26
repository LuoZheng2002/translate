from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env.private")  # reads .env into environment
api_key = os.getenv("DEEPSEEK_API_KEY")

with open("bfcl/bfcl_eval/data/BFCL_v4_multiple.json", "r") as f:
    lines = [line for line in f if line.strip()]

for i in range(len(lines)):
    lines[i] = json.loads(lines[i])
    # print(lines[i])
    # print(lines[i]['question'][0][0]['content'])

# Initialize client (API key can be set as env variable or directly)
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

system_message = {"role": "system", "content": 
         "你是一个翻译助手，请将用户提的问题翻译成中文，不要回答问题，不要换行。"
         }

# processed_lines = []
for (i, line) in enumerate(lines):
    line_content = line['question'][0][0]['content']  # Extract the content to be translated
    user_message = {"role": "user", "content": line_content}    
    # Create a chat completion
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            system_message,
            user_message
        ]
    )
    # Print the model’s reply
    processed_line = response.choices[0].message.content
    print(processed_line)
    lines[i]['question'][0][0]['content'] = processed_line
    

with open("BFCL_v4_multiple_zh_q.json", "w") as f:
    f.writelines(json.dumps(line, ensure_ascii=False) + '\n' for line in lines)
