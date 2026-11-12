import json
import os

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Study Planning Agent")
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

INSTRUCTIONS = """
You are a concise study-planning assistant.
Use the calculate_study_hours tool when the user asks for a total
based on days and hours per day. Explain the result plainly.
"""

TOOLS = [
    {
        "type": "function",
        "name": "calculate_study_hours",
        "description": "Calculate total study hours from days and hours per day.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of study days."
                },
                "hours_per_day": {
                    "type": "number",
                    "description": "Study hours per day."
                }
            },
            "required": ["days", "hours_per_day"],
            "additionalProperties": False
        },
        "strict": True
    }
]


def calculate_study_hours(days: int, hours_per_day: float) -> dict:
    # Validate in application code too. Never rely only on model output.
    if not 1 <= days <= 90:
        raise ValueError("days must be between 1 and 90")
    if not 0 < hours_per_day <= 16:
        raise ValueError("hours_per_day must be greater than 0 and at most 16")

    return {
        "days": days,
        "hours_per_day": hours_per_day,
        "total_hours": round(days * hours_per_day, 2)
    }


def run_agent(user_message: str) -> str:
    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=user_message,
        tools=TOOLS,
    )

    # Allow a few rounds in case the model needs more than one tool call.
    for _ in range(3):
        calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not calls:
            return response.output_text

        tool_outputs = []

        for call in calls:
            if call.name != "calculate_study_hours":
                raise ValueError(f"Unknown tool: {call.name}")

            print(f"Tool called: {call.name} with {call.arguments}")
            args = json.loads(call.arguments)
            result = calculate_study_hours(**args)

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })

        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOLS,
        )

    raise RuntimeError("Agent reached its tool-call limit")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = run_agent(request.message)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))