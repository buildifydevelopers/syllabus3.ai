from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import json

app = FastAPI(title="SyllabusAI", version="1.0.0")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

MODEL = "meta/llama-3.3-70b-instruct"  # change to any NIM model you have access to


# ── Models ──────────────────────────────────────────────────────────────────

class SyllabusInput(BaseModel):
    syllabus: str

class ParsedTopics(BaseModel):
    topics: List[str]

class ExplainInput(BaseModel):
    syllabus: str
    current_topic: str

class ExplainOutput(BaseModel):
    topic: str
    explanation: str
    next_topic: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


# ── Endpoint 1: Parse syllabus → clean topic list ───────────────────────────

@app.post("/parse-syllabus", response_model=ParsedTopics)
def parse_syllabus(data: SyllabusInput):
    """
    Raw messy syllabus in → clean list of study topics out.
    Strips unit headers, module labels, garbage text.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a syllabus parser. "
                    "Return ONLY valid JSON, no markdown, no extra text."
                ),
            },
            {
                "role": "user",
                "content": f"""Extract only the actual study topics from this raw syllabus.

Rules:
- Remove unit/module/chapter headers (Unit 1, Module A, Chapter 2, etc.)
- Remove garbage/noise text (random characters)
- Keep only real teachable topic names
- Return JSON: {{"topics": ["topic1", "topic2", ...]}}

Raw syllabus:
{data.syllabus}""",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content
    try:
        parsed = extract_json(raw)
        topics = parsed.get("topics", [])
        if not isinstance(topics, list):
            raise ValueError("topics not list")
        return ParsedTopics(topics=topics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse failed: {e} | Raw: {raw}")


# ── Endpoint 2: Explain topic + return next topic ────────────────────────────

@app.post("/explain-topic", response_model=ExplainOutput)
def explain_topic(data: ExplainInput):
    """
    Syllabus + current topic in → detailed explanation + next topic name out.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert teacher. "
                    "Return ONLY valid JSON, no markdown, no extra text."
                ),
            },
            {
                "role": "user",
                "content": f"""Given the syllabus and current topic, do two things:
1. Explain the current topic in detail with examples.
2. Find the next topic after the current one in the syllabus.

Return JSON with these keys:
- "topic": current topic name (string)
- "explanation": detailed explanation with examples, use \\n for newlines (string)
- "next_topic": next topic name (string) or null if last topic

Full syllabus:
{data.syllabus}

Current topic:
{data.current_topic}""",
            },
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content
    try:
        parsed = extract_json(raw)
        return ExplainOutput(
            topic=parsed["topic"],
            explanation=parsed["explanation"],
            next_topic=parsed.get("next_topic"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse failed: {e} | Raw: {raw}")


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "app": "SyllabusAI", "model": MODEL}
