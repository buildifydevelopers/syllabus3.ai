# ============================================================
#  EduPlatform Backend  -  Production Ready
#  Flow: parse syllabus -> generate plan -> study with chat
#  Chat: sends full syllabus + current topic, returns reply
#        + next_topic (locked on doubt, advances on mastery)
# ============================================================

import json
import re
import logging
from typing import Any, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EduPlatform")


class Settings(BaseSettings):
    nvidia_api_key: str = ""
    allowed_origin: str = "*"
    nvidia_model: str = "meta/llama-3.1-8b-instruct"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# Core AI helper
# ---------------------------------------------------------------------------
def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }


async def _chat_text(
    messages: list, max_tokens: int = 1024, temperature: float = 0.2
) -> str:
    payload = {
        "model": settings.nvidia_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(NVIDIA_API_URL, headers=_headers(), json=payload)
    if resp.status_code != 200:
        logger.error("NVIDIA API %s: %s", resp.status_code, resp.text[:300])
        raise HTTPException(status_code=502, detail="AI engine error")
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _clean_json(text: str) -> str:
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    # Extract outermost {...}
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([\]\}])", r"\1", text)
    return text


def _flatten_topics(data: Any) -> List[str]:
    """Recursively extract all non-trivial strings from any nested structure."""
    results: List[str] = []
    if isinstance(data, str):
        clean = data.strip()
        if len(clean) > 2:
            results.append(clean)
    elif isinstance(data, list):
        for item in data:
            results.extend(_flatten_topics(item))
    elif isinstance(data, dict):
        for val in data.values():
            results.extend(_flatten_topics(val))
    return results


def _dedup(lst: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str


class SyllabusRequest(BaseModel):
    subject: str
    raw_text: Optional[str] = ""


class SyllabusResponse(BaseModel):
    topics: List[str]
    estimated_total_hours: float
    topic_details: List[dict]


class PlanRequest(BaseModel):
    subject: str
    topics: List[str]
    start_date: str


class PlanResponse(BaseModel):
    schedule: list
    summary: str
    total_days: int
    daily_hours_recommended: float


class LectureChatRequest(BaseModel):
    subject: str
    topic: str
    full_syllabus: List[str] = []
    history: List[Message] = []
    message: str


class LectureChatResponse(BaseModel):
    reply: str
    next_topic: str
    progress_pct: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="EduPlatform AI", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.nvidia_model}


# ---------------------------------------------------------------------------
# 1. Parse Syllabus
# ---------------------------------------------------------------------------
@app.post("/syllabus/parse", response_model=SyllabusResponse)
async def parse_syllabus(req: SyllabusRequest):
    logger.info("parse_syllabus subject=%s text_len=%d", req.subject, len(req.raw_text or ""))

    prompt_schema = (
        '{"topics": ["Topic A", "Topic B"], '
        '"estimated_total_hours": 20.0, '
        '"topic_details": [{"topic": "Topic A", "estimated_min": 60, "complexity": "Medium"}]}'
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict academic topic extractor. "
                "Output ONLY valid JSON matching the schema given by the user. "
                "No extra text. No markdown. No explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract every academic topic/chapter from this {req.subject} syllabus.\n"
                f"Output exactly this JSON schema (flat list for topics, no nesting):\n"
                f"{prompt_schema}\n\n"
                f"SYLLABUS TEXT:\n{(req.raw_text or '')[:4000]}"
            ),
        },
    ]

    try:
        raw = await _chat_text(messages, temperature=0.1, max_tokens=2048)
        data = json.loads(_clean_json(raw))
    except Exception as e:
        logger.error("Syllabus parse AI error: %s", e)
        raise HTTPException(status_code=500, detail=f"Syllabus parse failed: {e}")

    # Flatten + dedup topics regardless of AI output shape
    raw_topics = data.get("topics", [])
    final_topics = _dedup(_flatten_topics(raw_topics))

    if not final_topics:
        logger.error("No topics extracted. Raw AI: %s", raw[:500])
        raise HTTPException(status_code=422, detail="Could not extract topics from syllabus.")

    data["topics"] = final_topics
    data["topic_details"] = [
        {"topic": t, "estimated_min": 60, "complexity": "Medium"} for t in final_topics
    ]
    data["estimated_total_hours"] = float(
        data.get("estimated_total_hours") or len(final_topics) * 1.5
    )

    logger.info("parse_syllabus extracted %d topics", len(final_topics))
    return SyllabusResponse(**data)


# ---------------------------------------------------------------------------
# 2. Generate Study Plan
# ---------------------------------------------------------------------------
@app.post("/plan/generate", response_model=PlanResponse)
async def generate_plan(req: PlanRequest):
    logger.info("generate_plan subject=%s topics=%d", req.subject, len(req.topics))

    prompt_schema = (
        '{"schedule": [{"day": 1, "date": "2026-05-09", "topic": "Topic A", "duration_mins": 60, "notes": "Focus on basics"}], '
        '"summary": "Study plan summary here.", '
        '"total_days": 10, '
        '"daily_hours_recommended": 2.0}'
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict study planner. "
                "Output ONLY valid JSON matching the schema given by the user. "
                "No extra text. No markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create a study plan for {req.subject} starting {req.start_date}.\n"
                f"Topics to cover: {req.topics}\n"
                f"Output exactly this JSON schema:\n{prompt_schema}"
            ),
        },
    ]

    try:
        raw = await _chat_text(messages, temperature=0.1, max_tokens=3000)
        data = json.loads(_clean_json(raw))
    except Exception as e:
        logger.error("Plan generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {e}")

    # Safety: ensure schedule is a list
    schedule = data.get("schedule", [])
    if isinstance(schedule, dict):
        schedule = list(schedule.values())

    return PlanResponse(
        schedule=schedule,
        summary=str(data.get("summary", "Study plan generated.")),
        total_days=int(data.get("total_days", len(schedule))),
        daily_hours_recommended=float(data.get("daily_hours_recommended", 2.0)),
    )


# ---------------------------------------------------------------------------
# 3. Lecture Chat  (Syllabus-Aware + Doubt Detection)
# ---------------------------------------------------------------------------
DOUBT_KEYWORDS = {"?", "confused", "dont understand", "don't understand", "not clear", "what is", "explain", "clarify"}


def _is_doubt(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in DOUBT_KEYWORDS)


@app.post("/lecture/chat", response_model=LectureChatResponse)
async def lecture_chat(req: LectureChatRequest):
    logger.info(
        "lecture_chat subject=%s topic=%s syllabus_len=%d",
        req.subject, req.topic, len(req.full_syllabus),
    )

    syllabus_str = " | ".join(req.full_syllabus)

    # Determine current index and total for progress calc
    try:
        current_idx = req.full_syllabus.index(req.topic)
    except ValueError:
        current_idx = 0
    total = max(len(req.full_syllabus), 1)

    system_content = (
        f"You are an expert academic tutor. Subject: {req.subject}.\n"
        f"Full Syllabus (in order): [{syllabus_str}]\n"
        f"Current Topic: {req.topic}\n\n"
        "RULES:\n"
        "1. Teach ONLY the current topic in depth.\n"
        "2. If the student asks a doubt or is confused, explain more on the SAME topic.\n"
        "3. If the student has understood and is ready to move on, set next_topic to the next topic in the syllabus.\n"
        "4. NEVER skip topics. NEVER make up topics outside the syllabus.\n"
        "5. Calculate progress_pct as (current_index / total_topics) * 100.\n"
        "6. Output ONLY this JSON, no extra text:\n"
        '{"reply": "your teaching text here", "next_topic": "exact topic name from syllabus", "progress_pct": 45.0}'
    )

    messages = [{"role": "system", "content": system_content}]
    for m in req.history[-8:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    try:
        raw = await _chat_text(messages, temperature=0.5, max_tokens=1024)
        data = json.loads(_clean_json(raw))
    except Exception as e:
        logger.error("Lecture chat error: %s | raw: %s", e, raw[:300] if "raw" in dir() else "")
        raise HTTPException(status_code=500, detail=f"Lecture chat failed: {e}")

    # Doubt lock: if user is asking a doubt, force next_topic = current topic
    if _is_doubt(req.message):
        next_topic = req.topic
        logger.info("Doubt detected - locking next_topic to: %s", next_topic)
    else:
        ai_next = data.get("next_topic", req.topic)
        # Validate: next_topic must be from the syllabus or same as current
        if ai_next in req.full_syllabus:
            next_topic = ai_next
        else:
            next_topic = req.topic

    progress_pct = float(data.get("progress_pct", round((current_idx / total) * 100, 1)))

    return LectureChatResponse(
        reply=data.get("reply", "Let me explain that for you."),
        next_topic=next_topic,
        progress_pct=progress_pct,
    )
