from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


if __name__ == "__main__":
    print(tokenizer.chat_template)