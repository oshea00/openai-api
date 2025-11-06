"""
Completion Models Comparison: Non-Reasoning vs Reasoning

This script demonstrates and compares "one shot" prompts between:
1. A non-reasoning model (GPT-4.1-mini)
2. A reasoning model (GPT-5-mini) configured for expedient responses

The purpose is to show how to configure reasoning models for optimal speed while
still leveraging their enhanced capabilities. The reasoning model is set up with:
- verbosity="low" to minimize output overhead
- reasoning_effort="minimal" to reduce thinking time

This comparison helps understand the trade-offs between model types and how to
tune reasoning models when speed is a priority while maintaining quality.
"""

from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env file
# OPENAI_API_KEY=jwt
# OPENAI_BASE_URL=
load_dotenv()

client = OpenAI()


def get_completion_4(messages: list[dict]) -> str:
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


def main():
    """
    Main function to execute completion demonstrations.
    Each demonstration is wrapped in a try-catch block to ensure
    that errors in one example don't stop the execution of others.
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

    print("=== GPT-4.1-mini Completion ===")
    try:
        response = get_completion_4(messages=test_messages)
        print(response)
        print()
    except Exception as e:
        print(f"❌ Error in get_completion_4: {e}")
        print()

    print("=== GPT-5-mini One-shot Completion ===")
    try:
        response = get_completion_5_oneshot(messages=test_messages)
        print(response)
        print()
    except Exception as e:
        print(f"❌ Error in get_completion_5_oneshot: {e}")
        print()


if __name__ == "__main__":
    main()
