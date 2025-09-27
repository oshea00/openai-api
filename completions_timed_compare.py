from openai import OpenAI
import time

client = OpenAI()


def get_completion_4o(messages: list[dict]) -> str:
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

# Measure time for get_completion_4o
start_4o = time.time()
for _ in range(4):
    response_4o = get_completion_4o(messages=test_messages)
end_4o = time.time()
total_4o_ms = int((end_4o - start_4o) * 1000)
print(response_4o)
print(f"Total execution time for get_completion_4o (4 runs): {total_4o_ms} ms")
print()
# Measure time for get_completion_5_oneshot
start_5 = time.time()
for _ in range(4):
    response_5 = get_completion_5_oneshot(messages=test_messages)
end_5 = time.time()
total_5_ms = int((end_5 - start_5) * 1000)
print(response_5)
print(f"Total execution time for get_completion_5_oneshot (4 runs): {total_5_ms} ms")
