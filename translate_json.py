from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env.private")  # reads .env into environment
api_key = os.getenv("OPENAI_API_KEY")

with open("bfcl/bfcl_eval/data/BFCL_v4_multiple.json", "r") as f:
    lines = [line for line in f if line.strip()]

# Initialize client (API key can be set as env variable or directly)
client = OpenAI(api_key=api_key)  # or omit api_key if you set it in the environment

system_message = {"role": "system", "content": 
         "You are a helpful assistant. Please translate some parts of the user specified json string into Chinese and keep the rest unchanged. "
         "The parts to be translated are: content, function name and description, parameter name and description. For function names, treat '_' as ' ' and then translate."
         " Do not translate the json keys except for parameter names. Do not translate python literals such as dict, string, true, false."
         "Do not translate the punctuation marks, brackets, braces, commas, colons, and quotation marks.\n"
         "Here is an example of a json string to be translated:\n"
         '{"id": "multiple_2", "question": [[{"role": "user", "content": "What is the capital of Brazil?"}]], '
         '"function": [{"name": "country_info.largest_city", "description": "Fetch the largest city of a specified country.", '
         '"parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, '
         '"required": ["country"]}}, {"name": "country_info.capital", "description": "Fetch the capital city of a specified country.", '
         '"parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, '
         '"required": ["country"]}}, {"name": "country_info.population", "description": '
         '"Fetch the current population of a specified country.", '
         '"parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, '
         '"required": ["country"]}}]}\n\n'
         "The correct translation is:\n"
         '{"id": "multiple_2", "question": [[{"role": "user", "content": "巴西的首都是什么?"}]], '
         '"function": [{"name": "国家信息.最大的城市", "description": "获取指定国家的最大城市.", '
         '"parameters": {"type": "dict", "properties": {"国家": {"type": "string", "description": "国家名称."}}, '
         '"required": ["国家"]}}, {"name": "国家信息.首都", "description": "获取指定国家的首都.", '
         '"parameters": {"type": "dict", "properties": {"国家": {"type": "string", "description": "国家名称."}}, '
         '"required": ["国家"]}}, {"name": "国家信息.人口", "description": "获取指定国家的当前人口.", '
         '"parameters": {"type": "dict", "properties": {"国家": {"type": "string", "description": "国家名称."}}, '
         '"required": ["国家"]}}]}\n\n'
         'Notice how the function names and parameters are also translated although they may not be valid programming language identifiers.'
         }

processed_lines = []
for line in lines:
    user_message = {"role": "user", "content": line}    
    # Create a chat completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            system_message,
            user_message
        ]
    )
    # Print the model’s reply
    processed_line = response.choices[0].message.content
    processed_lines.append(processed_line)
    print(processed_line)

with open("bfcl/bfcl_eval/data/BFCL_v4_multiple_zh.json", "w") as f:
    f.writelines(processed_lines)
