from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


def basic_text_chat(question):
    response = client.responses.create(model="gpt-5", input=question)
    print(response.output_text)


def structured_response_model(model, question):
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Extract the event information."},
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

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
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
            }
        },
    )
    print(response.output_text)


def response_with_reasoning():
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
        ],
        reasoning={"effort": "medium", "summary": "auto"},
    )
    print(response.output_text)
    print("Summary:")
    summary = ""
    for r in response.output:
        if r.type == "reasoning":
            for s in r.summary:
                summary += s.text + " "
    print(summary)


print("=== Basic Text Chat ===")
basic_text_chat("Write a one-sentence bedtime story about a unicorn.")

print("\n=== Structured Response Model ===")
structured_response_model(
    "gpt-5", "Create a calendar event for a meeting with Alice and Bob on July 24th."
)

print("\n=== Structured Response JSON Mode ===")
structured_response_json_mode("Alice and Bob are meeting on July 24th, 2025.")

print("\n=== Structured Response Text ===")
structure_response_text()

print("\n=== Response with Reasoning ===")
response_with_reasoning()
