"""
Responses API: Structured Responses and Reasoning Capabilities

This script demonstrates various approaches to get structured responses and reasoning
capabilities from the OpenAI Responses API. It serves as a counterpart to the legacy
chat completions API, showcasing the newer response formats and enhanced features
available through the modern responses interface.

Key demonstrations include:
1. Basic text responses - Simple conversational responses using the responses API
2. Structured responses using Pydantic models (responses parse endpoint)
3. JSON mode responses with traditional chat completions (for comparison)
4. Strict JSON schema enforcement using the responses API text format
5. Reasoning capabilities with configurable effort levels and automatic summaries

Responses API Advantages:
- Native support for reasoning with configurable effort levels
- Built-in reasoning summary extraction capabilities
- Streamlined structured output with text_format parameter
- Enhanced schema validation and parsing

Debugging Features:
- Custom LoggingHTTPClient that intercepts and logs all HTTP requests/responses
- Shows exactly how the OpenAI library constructs API calls behind the scenes
- Displays request headers, body, response status, and full response data
- Masks sensitive authorization tokens for security

This provides both functional examples of the modern responses API patterns and visibility
into the underlying HTTP communication for debugging and learning purposes.
"""

from openai import OpenAI
from pydantic import BaseModel
import json
import httpx
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
# OPENAI_API_KEY=jwt
# OPENAI_BASE_URL=
load_dotenv()


class LoggingHTTPClient(httpx.Client):
    """
    Custom HTTP client that logs all requests and responses for debugging.
    """

    def send(self, request, **kwargs):
        print("=" * 80)
        print("🔍 REQUEST DETAILS:")
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print("Headers:")
        for name, value in request.headers.items():
            # Mask authorization header for security
            if name.lower() == "authorization":
                print(f"  {name}: Bearer ***masked***")
            else:
                print(f"  {name}: {value}")

        if request.content:
            try:
                # Pretty print JSON content
                content = request.content.decode("utf-8")
                parsed_json = json.loads(content)
                print("Request Body:")
                print(json.dumps(parsed_json, indent=2))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"Request Body (raw): {request.content}")

        print("-" * 40)

        # Send the actual request
        response = super().send(request, **kwargs)

        print("📥 RESPONSE DETAILS:")
        print(f"Status Code: {response.status_code}")
        print("Response Headers:")
        for name, value in response.headers.items():
            print(f"  {name}: {value}")

        try:
            # Pretty print JSON response
            response_json = response.json()
            print("Response Body:")
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print(f"Response Body (raw): {response.text}")

        print("=" * 80)
        print()

        return response


# Create OpenAI client with custom HTTP client for request logging
client = OpenAI(http_client=LoggingHTTPClient())


def basic_text_chat(question):
    """
    Demonstrates basic response creation with the responses API.
    Uses the gpt-5 model to generate a simple text response.
    """
    response = client.responses.create(model="gpt-5", input=question)
    print(response.output_text)


def structured_response_model(model, question):
    """
    Demonstrates structured output using Pydantic models with the responses parse endpoint.
    The model extracts information and returns it as a validated Pydantic object.
    """

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": "Extract the event information.",
            },
            {"role": "user", "content": question},
        ],
        text_format=CalendarEvent,
        reasoning={
            "effort": "minimal",
        },
    )

    event = response.output_parsed
    print(event)
    print(response.output_text)


def structured_response_json_mode(question):
    """
    Demonstrates JSON mode where the model is constrained to return valid JSON.
    The structure is defined in the system prompt rather than enforced by schema.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract the event information as json with keys name, date, participants.",
            },
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )

    print(response.choices[0].message.content)


def structure_response_text():
    """
    Demonstrates strict JSON schema enforcement using responses API with structured format.
    The model output is validated against the provided schema with strict mode enabled.
    """
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {
                "role": "user",
                "content": "how can I solve 8x + 7 = -23",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "math_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"},
                                },
                                "required": [
                                    "explanation",
                                    "output",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "final_answer": {"type": "string"},
                    },
                    "required": [
                        "steps",
                        "final_answer",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
    )
    print(response.output_text)


def extract_reasoning_summary(response):
    """
    Extracts reasoning summary from a response object.
    Returns a concatenated string of all reasoning summary text.
    """
    return " ".join(
        s.text for r in response.output if r.type == "reasoning" for s in r.summary
    )


def response_with_reasoning():
    """
    Demonstrates the responses API with reasoning capabilities enabled.
    Shows how to extract and display both the main response and reasoning summary.
    """
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {
                "role": "user",
                "content": "how can I solve 8x + 7 = -23",
            },
        ],
        reasoning={
            "effort": "medium",
            "summary": "auto",
        },
    )
    print(response.output_text)
    print("Summary:")

    print(extract_reasoning_summary(response))


def main():
    """
    Main function to execute all the example demonstrations.
    Each demonstration is wrapped in a try-catch block to ensure
    that errors in one example don't stop the execution of others.
    """
    print("=== Basic Text Chat ===")
    try:
        basic_text_chat("Write a one-sentence bedtime story about a unicorn.")
    except Exception as e:
        print(f"❌ Error in basic_text_chat: {e}")
        print()

    print("\n=== Structured Response Model ===")
    try:
        structured_response_model(
            "gpt-5",
            "Create a calendar event for a meeting with Alice and Bob on July 24th.",
        )
    except Exception as e:
        print(f"❌ Error in structured_response_model: {e}")
        print()

    print("\n=== Structured Response JSON Mode ===")
    try:
        structured_response_json_mode("Alice and Bob are meeting on July 24th, 2025.")
    except Exception as e:
        print(f"❌ Error in structured_response_json_mode: {e}")
        print()

    print("\n=== Structured Response Text ===")
    try:
        structure_response_text()
    except Exception as e:
        print(f"❌ Error in structure_response_text: {e}")
        print()

    print("\n=== Response with Reasoning ===")
    try:
        response_with_reasoning()
    except Exception as e:
        print(f"❌ Error in response_with_reasoning: {e}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAI Responses API demonstrations with optional logging to file"
    )
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        help="Optional log file to write output to instead of console",
    )

    args = parser.parse_args()

    # Redirect output to file if specified
    if args.log_file:
        original_stdout = sys.stdout
        try:
            with open(args.log_file, "w", encoding="utf-8") as log_file:
                sys.stdout = log_file
                main()
        except Exception as e:
            sys.stdout = original_stdout
            print(f"Error writing to log file '{args.log_file}': {e}")
            sys.exit(1)
        finally:
            sys.stdout = original_stdout
        print(f"Output written to: {args.log_file}")
    else:
        main()
