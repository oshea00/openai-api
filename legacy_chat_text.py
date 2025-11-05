from openai import OpenAI
from pydantic import BaseModel
import json
import httpx
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
        print(f"Headers:")
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
                print(f"Request Body:")
                print(json.dumps(parsed_json, indent=2))
            except (UnicodeDecodeError, json.JSONDecodeError):
                print(f"Request Body (raw): {request.content}")

        print("-" * 40)

        # Send the actual request
        response = super().send(request, **kwargs)

        print(f"📥 RESPONSE DETAILS:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers:")
        for name, value in response.headers.items():
            print(f"  {name}: {value}")

        try:
            # Pretty print JSON response
            response_json = response.json()
            print(f"Response Body:")
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
    Demonstrates basic chat completion with a simple user message.
    The model generates a text response without any special formatting or tools.
    """
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": question}]
    )
    print(response.choices[0].message.content)


def structured_response_model(model, question):
    """
    Demonstrates structured output using Pydantic models with the beta parse endpoint.
    The model extracts information and returns it as a validated Pydantic object.
    """

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": question},
        ],
        response_format=CalendarEvent,
    )

    event = response.choices[0].message.parsed
    print(event)
    print(response.choices[0].message.content)


def structured_response_json_mode(question):
    """
    Demonstrates JSON mode where the model is constrained to return valid JSON.
    The structure is defined in the system prompt rather than enforced by schema.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
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
    Demonstrates strict JSON schema enforcement using json_schema response format.
    The model output is validated against the provided schema with strict mode enabled.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
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
                                "required": ["explanation", "output"],
                                "additionalProperties": False,
                            },
                        },
                        "final_answer": {"type": "string"},
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )
    print(response.choices[0].message.content)


def get_weather(city, country):
    """Simulated weather tool that takes discrete parameters"""
    # Simulate weather data
    return {
        "location": f"{city}, {country}",
        "temperature": "72°F",
        "conditions": "Partly cloudy",
        "humidity": "65%",
    }


def tools_call_example():
    """
    Implements the standard two-phase tool calling pattern:

    Phase 1 - Initial Request:
    1. The model receives the user query and tool definitions
    2. If the model determines a tool is needed, it returns a tool_calls object (not actual text)
    3. The response contains the function name and JSON arguments to call

    Phase 2 - Tool Execution & Final Response:
    1. The original assistant message (with tool_calls) is appended to messages
    2. Each tool is executed locally with the extracted arguments
    3. Tool results are added to messages with role: "tool" and the tool_call_id linking them to the original call
    4. A second LLM call is made with the full conversation history, including tool results
    5. The model now generates natural language incorporating the tool data

    The second call is necessary because the first call only produces structured tool calls, not a user-facing response.
    The model needs to see the actual tool results to formulate its final answer.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "country": {
                            "type": "string",
                            "description": "The country name",
                        },
                    },
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco, USA?"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            print(f"Calling {function_name}...")
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "get_weather":
                function_response = get_weather(
                    city=function_args["city"], country=function_args["country"]
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    }
                )

        second_response = client.chat.completions.create(
            model="gpt-4o", messages=messages
        )

        print(second_response.choices[0].message.content)
    else:
        print(response_message.content)


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
            "gpt-4o",
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

    print("\n=== Tools Call Example ===")
    try:
        tools_call_example()
    except Exception as e:
        print(f"❌ Error in tools_call_example: {e}")
        print()


if __name__ == "__main__":
    main()
