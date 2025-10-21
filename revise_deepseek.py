import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.private")  # reads .env into environment
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

with open("chinese_processed.json", "r") as f:
    lines = [line for line in f if line.strip()]


system_message = {"role": "system", "content": 
        "你是一个校对助手。用户将输入两个json字符串，请检查第二个json字符串中ground_truth中的函数名，参数名和的参数值是否在第一个json字符串中出现过（相同语言，完全匹配）。"
        "如果发现ground_truth中的内容与第一个json字符串中的内容语义相近，但语言不同或不完全字字对应，请将对应的内容修正。"
        "特别地，如果ground_truth中的可接受的参数列表与第一个json字符串中的内容无法完全匹配，则将缺失的选项补充到json数组中。\n"
        "如果遇到空字符串\"\"，请将其保留。\n"
        "除此之外，如果第一个json字符串的问题表述中有关键词与函数调用使用的关键词不一致，也请进行修正。\n"
        "直接输出答案，以‘{’开头，不要使用markdown格式。输出应包含与输入相同的两个json字符串。除每个json字符串末尾添加换行符外，不要添加换行或缩进。如果没有需要修改的地方，则直接输出原内容。\n"
        "如果有不确定的地方，在答案后另起一行进行说明。\n"
        "以下是一个需要修改的例子：\n\n"
        '{"id": "multiple_147", "question": [[{"role": "user", "content": '
        '"给我从纽约到洛杉矶避开高速公路和收费公路的路线。"}]], '
        '"function": [{"name": "地图服务.获取路线", "description": '
        '"获取从起点到终点的路线,包括路线偏好选项.", "parameters": {"type": "dict", "properties": '
        '{"起点": {"type": "string", "description": "路线的起始位置."}, "终点": {"type": "string", '
        '"description": "路线的结束位置."}, "避开": {"type": "array", "items": {"type": "string", '
        '"enum": ["tolls", "highways", "ferries"]}, "description": "要避开的路线特征.默认为空数组."}}, '
        '"required": ["起点", "终点"]}}, {"name": "转换货币", "description": "将金额从特定货币转换为另一种货币.", '
        '"parameters": {"type": "dict", "properties": {"基础货币": {"type": "string", "description": "原始金额所在的基础货币."}, '
        '"目标货币": {"type": "string", "description": "要转换到的货币."}, "金额": {"type": "integer", "description": '
        '"要转换的金额."}}, "required": ["基础货币", "目标货币", "金额"]}}, {"name": "生态学.获取海龟种群", "description": '
        '"获取特定位置的海龟种群和物种.", "parameters": {"type": "dict", "properties": {"位置": {"type": "string", '
        '"description": "位置名称."}, "年份": {"type": "integer", "description": "请求数据的年份.(可选)默认为2024."}, '
        '"物种": {"type": "boolean", "description": "是否包含物种信息.默认为false.(可选)"}}, "required": ["位置"]}}]}\n'
        '{"id": "multiple_147", "ground_truth": [{"地图服务.获取路线": {"起点": ["New York", "New York, NY", "NYC"], '
        '"终点": ["Los Angeles", "LA"], "避开": [["highways", "tolls"], ["tolls", "highways"]]}}]}\n\n'
        '修改思路：注意到问题中有“高速公路”和“收费公路”两个关键词，而提供的函数的可选参数有["tolls", "highways", "ferries"]，与问题中的关键词语言不一致。'
        '此外，问题中提到“纽约”和“洛杉矶”，而ground_truth中的参数值为["New York", "New York, NY", "NYC"]和["Los Angeles", "LA"]，而没有中文版的“纽约”和“洛杉矶”。'
        '因此正确的修改结果应为：\n\n'
        '{"id": "multiple_147", "question": [[{"role": "user", "content": '
        '"给我从纽约到洛杉矶避开高速公路和收费公路的路线。"}]], '
        '"function": [{"name": "地图服务.获取路线", "description": '
        '"获取从起点到终点的路线,包括路线偏好选项.", "parameters": {"type": "dict", "properties": '
        '{"起点": {"type": "string", "description": "路线的起始位置."}, "终点": {"type": "string", '
        '"description": "路线的结束位置."}, "避开": {"type": "array", "items": {"type": "string", '
        '"enum": ["收费公路", "高速公路", "渡船"]}, "description": "要避开的路线特征.默认为空数组."}}, '
        '"required": ["起点", "终点"]}}, {"name": "转换货币", "description": "将金额从特定货币转换为另一种货币.", '
        '"parameters": {"type": "dict", "properties": {"基础货币": {"type": "string", "description": "原始金额所在的基础货币."}, '
        '"目标货币": {"type": "string", "description": "要转换到的货币."}, "金额": {"type": "integer", "description": '
        '"要转换的金额."}}, "required": ["基础货币", "目标货币", "金额"]}}, {"name": "生态学.获取海龟种群", "description": '
        '"获取特定位置的海龟种群和物种.", "parameters": {"type": "dict", "properties": {"位置": {"type": "string", '
        '"description": "位置名称."}, "年份": {"type": "integer", "description": "请求数据的年份.(可选)默认为2024."}, '
        '"物种": {"type": "boolean", "description": "是否包含物种信息.默认为false.(可选)"}}, "required": ["位置"]}}]}\n'
        '{"id": "multiple_147", "ground_truth": [{"地图服务.获取路线": {"起点": ["New York", "New York, NY", "NYC", "纽约"], '
        '"终点": ["Los Angeles", "LA", "洛杉矶"], "避开": [["高速公路", "收费公路"], ["收费公路", "高速公路"]]}}]}\n'
        }

processed_lines = []
for i, (line1, line2) in enumerate(zip(lines[::2], lines[1::2])):
    if i <=33:
        continue  # skip already processed lines
    user_message = {"role": "user", "content": line1 + '\n' + line2}    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            system_message,
            user_message
        ],
        stream=False
    )
    processed_line = response.choices[0].message.content
    processed_lines.append(processed_line)
    print(processed_line)
    with open("chinese_auto_revised.json", "w") as f:
        f.writelines(line + '\n' for line in processed_lines)
