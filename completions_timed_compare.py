"""
Timed Performance Comparison: Non-Reasoning vs Reasoning Models

This script performs a quantitative performance comparison between non-reasoning
and reasoning models to measure the actual time overhead of reasoning capabilities.

The test runs the same prompt 4 times on each model:
- GPT-4.1-mini (non-reasoning): baseline performance
- GPT-5-mini (reasoning): configured with minimal reasoning effort

Expected outcome: The reasoning model takes longer overall than the non-reasoning
model, even when configured for speed (low verbosity, minimal reasoning effort).
This quantifies the performance cost of reasoning capabilities.

The timing data helps inform decisions about when to use reasoning models vs
traditional models based on performance requirements and use case priorities.
Results typically show reasoning models are slower but provide enhanced
problem-solving capabilities.
"""

from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env file
# OPENAI_API_KEY=jwt
# OPENAI_BASE_URL=
load_dotenv()

client = OpenAI()


def get_completion_4o(messages: list[dict]) -> str:
    """
    Demonstrates basic chat completion using GPT-4.1-mini model.
    Uses temperature=0 for deterministic responses.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        str: The message object from the model's response
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, temperature=0
    )
    return response.choices[0].message


def get_completion_5_oneshot(messages: list[dict]) -> str:
    """
    Demonstrates chat completion using GPT-5-mini model with reasoning capabilities.
    Uses low verbosity and minimal reasoning effort for faster responses.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        str: The message object from the model's response
    """
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        verbosity="low",
        reasoning_effort="minimal",
    )
    return response.choices[0].message


def timed_comparison_test():
    """
    Performs a timed comparison between GPT-4.1-mini and GPT-5-mini models.
    Runs each model 4 times and measures total execution time.

    Returns:
        tuple: (response_4o, total_4o_ms, response_5, total_5_ms)
    """
    # Test messages for demonstrations
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

    # Measure time for get_completion_5_oneshot
    start_5 = time.time()
    for _ in range(4):
        response_5 = get_completion_5_oneshot(messages=test_messages)
    end_5 = time.time()
    total_5_ms = int((end_5 - start_5) * 1000)

    return response_4o, total_4o_ms, response_5, total_5_ms


def main():
    """
    Main function to execute timed completion comparison.
    Wraps the test in a try-catch block to handle potential errors gracefully.
    """
    print("=== Timed Completion Comparison ===")
    try:
        response_4o, total_4o_ms, response_5, total_5_ms = timed_comparison_test()

        print("GPT-4.1-mini Response:")
        print(response_4o)
        print(f"Total execution time for get_completion_4o (4 runs): {total_4o_ms} ms")
        print()

        print("GPT-5-mini Response:")
        print(response_5)
        print(
            f"Total execution time for get_completion_5_oneshot (4 runs): {total_5_ms} ms"
        )
        print()

        # Calculate and display performance comparison
        if total_4o_ms > 0 and total_5_ms > 0:
            speed_ratio = total_4o_ms / total_5_ms
            if speed_ratio > 1:
                print(f"GPT-5-mini is {speed_ratio:.2f}x faster than GPT-4.1-mini")
            else:
                print(f"GPT-4.1-mini is {1/speed_ratio:.2f}x faster than GPT-5-mini")

    except Exception as e:
        print(f"❌ Error in timed_comparison_test: {e}")
        print()


if __name__ == "__main__":
    main()
