from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env file
# OPENAI_API_KEY=jwt
# OPENAI_BASE_URL=
load_dotenv()

client = OpenAI()


def get_completion_4(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0
    )
    return response.choices[0].message


def get_completion_5_oneshot(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        verbosity="low",
        reasoning_effort="minimal",
    )
    return response.choices[0].message


test_messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": "say hello and comment on the weather.",
    },
]

response = get_completion_5_oneshot(messages=test_messages)
print(response)
print()
