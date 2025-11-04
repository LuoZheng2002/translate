from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from enum import Enum, auto
from config import Language
import re
from parse_dataset import load_json_lines
# Load API keys from .env file
load_dotenv(dotenv_path=".env.private")


class TranslateOption(Enum):
    FULLY_TRANSLATED = auto()
    PARTIALLY_TRANSLATED = auto()

class TranslateConfig:
    def __init__(self, language: Language, option: TranslateOption):
        self.language = language
        self.option = option

# language_to_translate: list[Language] = [Language.CHINESE]

translate_configs: list[TranslateConfig] = [
    TranslateConfig(language=Language.CHINESE, option=TranslateOption.FULLY_TRANSLATED),
    TranslateConfig(language=Language.CHINESE, option=TranslateOption.PARTIALLY_TRANSLATED),
]

for config in translate_configs:
    print(f"Translating dataset for language: {config.language}, option: {config.option}")
    # === Choose which model to use ===
    # Options: "deepseek" or "openai"
    # MODEL_PROVIDER = "deepseek"  # change this to "openai" to switch

    # === Model and client configuration ===
    match config.language:
        case Language.CHINESE:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
            model_name = "deepseek-chat"
        case Language.HINDI:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
            model_name = "gpt-4o-mini"
        case _:
            raise ValueError("MODEL_PROVIDER must be either 'deepseek' or 'openai'")
    
    match config.language:
        case Language.CHINESE:
            translate_mode_postfix = "_zh"
        case Language.HINDI:
            translate_mode_postfix = "_hi"
    match config.option:
        case TranslateOption.FULLY_TRANSLATED:
            translate_mode_postfix += "_full"
        case TranslateOption.PARTIALLY_TRANSLATED:
            translate_mode_postfix += "_partial"
    output_path = f"dataset/BFCL_v4_multiple{translate_mode_postfix}.json"
    # Initialize the client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # === Load input file ===
    with open("dataset/BFCL_v4_multiple.json", "r", encoding="utf-8") as f:
        dataset = load_json_lines(f)
    with open("dataset/possible_answer/BFCL_v4_multiple.json", "r", encoding="utf-8") as f:
        possible_answers = load_json_lines(f)
    # === Process each line ===
    existing_indices = []
    translated_lines = []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            translated_lines = load_json_lines(f)
            existing_indices = [item["id"] for item in translated_lines]
    except FileNotFoundError:
        print(f"No existing translated dataset found at {output_path}. A new one will be created.")
    with open(output_path, "w", encoding="utf-8") as f_out:
        warning_printed = False
        for dataset_line in dataset:
            id = dataset_line["id"]
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {output_path}.")
                    warning_printed = True
                continue
            # === Translation system message ===
            match (config.language, config.option):
                case (Language.CHINESE, TranslateOption.FULLY_TRANSLATED):
                    system_message = {
                        "role": "system",
                        "content": "你是一个翻译助手，请将用户提的问题翻译成中文，不要回答问题，不要换行。"
                    }
                case (Language.CHINESE, TranslateOption.PARTIALLY_TRANSLATED):
                    possible_answer = next((ans for ans in possible_answers if ans["id"] == id), None)
                    assert possible_answer is not None, f"Possible answer not found for id {id}"
                    possible_answer = possible_answer['ground_truth']
                    possible_answer = json.dumps(possible_answer, ensure_ascii=False)
                    system_message = {
                        "role": "system",
                        "content": "你是一个翻译助手，请将用户提的问题翻译成中文，不要回答问题，不要换行。\n如果原文中的词语在以下json字符串的内容中出现，请保留这些词语不翻译。\n" + possible_answer
                    }
                case (Language.HINDI, TranslateOption.FULLY_TRANSLATED):
                    system_message = {
                        "role": "system",
                        "content": "You are a translation assistant. Please translate the user's question into Hindi. Do not answer the question. Do not add line breaks."
                    }
                case (Language.HINDI, TranslateOption.PARTIALLY_TRANSLATED):
                    possible_answer = next((ans for ans in possible_answers if ans["id"] == id), None)
                    assert possible_answer is not None, f"Possible answer not found for id {id}"
                    possible_answer = possible_answer['ground_truth']
                    possible_answer = json.dumps(possible_answer, ensure_ascii=False)
                    system_message = {
                        "role": "system",
                        "content": "You are a translation assistant. Please translate the user's question into Hindi. Do not answer the question. Do not add line breaks.\nIf any words from the following JSON string appear in the original text, please keep those words untranslated.\n" + possible_answer
                    }
            # === Prepare user message and call API ===

            line_content = dataset_line["question"][0][0]["content"]
            user_message = {"role": "user", "content": line_content}

            response = client.chat.completions.create(
                model=model_name,
                messages=[system_message, user_message],
            )

            translated = response.choices[0].message.content.strip()

            # Update the line content
            translated_line = dataset_line.copy()
            translated_line["question"][0][0]["content"] = translated
            translated_lines.append(translated_line)
            f_out.seek(0)
            f_out.truncate()
            for t_line in translated_lines:
                f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
            f_out.flush()
        # sort the lines
        translated_lines = sorted(translated_lines, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
        f_out.seek(0)
        f_out.truncate()
        for t_line in translated_lines:
            f_out.write(json.dumps(t_line, ensure_ascii=False) + "\n")
        f_out.flush()
    print(f"\n✅ Translation complete! Output saved to: {output_path}")

    # # === Save output file ===
    # output_filename = f"BFCL_v4_multiple_zh_q_{MODEL_PROVIDER}.json"
    # with open(output_filename, "w", encoding="utf-8") as f:
    #     for dataset_line in dataset:
    #         f.write(json.dumps(dataset_line, ensure_ascii=False) + "\n")

    # print(f"\n✅ Translation complete! Output saved to: {output_filename}")
