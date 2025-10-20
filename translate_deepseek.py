# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.private")  # reads .env into environment
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

with open("bfcl/bfcl_eval/data/BFCL_v4_multiple.json", "r") as f:
    prompt_lines = [line for line in f if line.strip()]

with open("bfcl/bfcl_eval/data/possible_answer/BFCL_v4_multiple.json", "r") as f:
    answer_lines = [line for line in f if line.strip()]

system_message = {"role": "system", "content": 
         "你是一个翻译助手。请将用户输入的json字符串的一些部分翻译成中文，其余部分保持不变。"
         "需要翻译的部分有：content， function name 和 description， parameter name 和 description. "
         "对于函数名字， 将'_'视为' '后再进行翻译，原则上不保留下划线，除非删除下划线会影响语义。如遇英文专有名词，如'id'或品牌名称，可不翻译，按照中文习惯处理即可。"
         "如遇到中文和英文混合的情况，如'建筑id'，中英文间不添加下划线或空格。"
         "函数名称和参数中不要带有空格，如实在需要分隔单词，可以使用下划线"
         "除了函数参数名字外，不要翻译json的键。不要翻译python字面量，如dict，string，true，false。"
         "保留英文风格的标点符号、括号、花括号、逗号、冒号和引号。\n"
         "保持翻译的一致性，即相同的英文单词应翻译成相同的中文。\n"
         "下面是一个需要翻译的json字符串的例子：\n"
         '{"id": "multiple_2", "question": [[{"role": "user", "content": "What is the capital of Brazil?"}]], '
         '"function": [{"name": "country_info.largest_city", "description": "Fetch the largest city of a specified country.", '
         '"parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, '
         '"required": ["country"]}}, {"name": "country_info.capital", "description": "Fetch the capital city of a specified country.", '
         '"parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, '
         '"required": ["country"]}}, {"name": "country_info.population", "description": '
         '"Fetch the current population of a specified country.", '
         '"parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, '
         '"required": ["country"]}}]}\n'
         '{"id": "multiple_2", "ground_truth": [{"country_info.capital": {"country": ["Brazil"]}}]}\n\n'
         "正确的翻译是：\n"
         '{"id": "multiple_2", "question": [[{"role": "user", "content": "巴西的首都是什么?"}]], '
         '"function": [{"name": "国家信息.最大的城市", "description": "获取指定国家的最大城市.", '
         '"parameters": {"type": "dict", "properties": {"国家": {"type": "string", "description": "国家名称."}}, '
         '"required": ["国家"]}}, {"name": "国家信息.首都", "description": "获取指定国家的首都.", '
         '"parameters": {"type": "dict", "properties": {"国家": {"type": "string", "description": "国家名称."}}, '
         '"required": ["国家"]}}, {"name": "国家信息.人口", "description": "获取指定国家的当前人口.", '
         '"parameters": {"type": "dict", "properties": {"国家": {"type": "string", "description": "国家名称."}}, '
         '"required": ["国家"]}}]}\n'
         '{"id": "multiple_2", "ground_truth": [{"国家信息.首都": {"国家": ["巴西"]}}]}\n\n'
         '注意函数名字和参数名字也被翻译了，尽管它们可能不是有效的编程语言标识符。\n'
         '回答仅保留翻译结果（第一个字符应是"{"，不要使用markdown格式），如遇无法翻译，翻译规则有冲突，或任何不确定的情形，在翻译后另起一行进行说明。'
         }

processed_lines = []
for prompt_line, answer_line in zip(prompt_lines, answer_lines):
    user_message = {"role": "user", "content": prompt_line + '\n' + answer_line}    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            system_message,
            user_message
        ],
        stream=False
    )
    # Print the model’s reply
    processed_line = response.choices[0].message.content
    processed_lines.append(processed_line)
    print(processed_line)
    with open("chinese.json", "w") as f:
        f.writelines(line + '\n' for line in processed_lines)

