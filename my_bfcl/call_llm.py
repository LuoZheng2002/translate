def gpt_4o_mini_inference(input_messages: list) -> str:
    from openai import OpenAI
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path=".env")  # reads .env into environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialize client (API key can be set as env variable or directly)
    client = OpenAI(api_key=api_key)  # or omit api_key if you set it in the environment

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=input_messages
    )
    processed_line = response.choices[0].message.content
    return processed_line
