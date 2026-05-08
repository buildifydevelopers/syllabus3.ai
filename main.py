# ============================================================
#  EduPlatform Backend � NVIDIA NIM Version (Optimized)
# ============================================================

import base64
import json
import re
import logging
from datetime import datetime as dt
from typing import List, Optional, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO)
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

def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.nvidia_api_key}", "Content-Type": "application/json"}

async def _chat_text(messages: list, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    payload = {"model": settings.nvidia_model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(NVIDIA_API_URL, headers=_headers(), json=payload)
    if resp.status_code != 200: raise HTTPException(status_code=502, detail="AI error")
    return resp.json()["choices"][0]["message"]["content"].strip()

def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match: text = match.group(1)
    return text

def flatten_topics(data: Any) -> List[str]:
    """Recursively extract all strings from nested lists or dicts."""
    results = []
    if isinstance(data, str):
        results.append(data)
    elif isinstance(data, list):
        for item in data:
            results.extend(flatten_topics(item))
    elif isinstance(data, dict):
        for val in data.values():
            results.extend(flatten_topics(val))
    return results

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
    history: List[Message]
    message: str

class LectureChatResponse(BaseModel):
    reply: str
    next_topic: Optional[str] = None
    progress_pct: float

app = FastAPI(title="EduPlatform AI")
app.add_middleware(CORSMiddleware, allow_origins=[settings.allowed_origin], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/syllabus/parse", response_model=SyllabusResponse)
async def parse_syllabus(req: SyllabusRequest):
    messages = [
        {"role": "system", "content": "You are a rigid Data Extraction Bot. Output ONLY valid JSON."},
        {"role": "user", "content": f"""Extract topics from {req.subject} syllabus.
Output JSON: {{"topics": ["Name"], "estimated_total_hours": 10.0, "topic_details": [{{"topic": "Name", "estimated_min": 60, "complexity": "Medium"}}]}}
TEXT: {req.raw_text[:4000]}"""}
    ]
    raw = await _chat_text(messages, temperature=0.1)
    data = json.loads(_clean_json(raw))
    
    # Powerful Flattening Logic
    all_topics = flatten_topics(data.get("topics", []))
    # Filter out empty or extremely short strings
    final_topics = [t for t in all_topics if len(t) > 2]
    
    # Reconstruct data structure
    data["topics"] = final_topics
    data["topic_details"] = [{"topic": t, "estimated_min": 60, "complexity": "Medium"} for t in final_topics]
    data["estimated_total_hours"] = data.get("estimated_total_hours", len(final_topics) * 1.5)
    
    return SyllabusResponse(**data)

@app.post("/plan/generate", response_model=PlanResponse)
async def generate_plan(req: PlanRequest):
    messages = [{"role": "system", "content": "Create study plan JSON."}, {"role": "user", "content": f"Subject: {req.subject}. Topics: {req.topics}"}]
    raw = await _chat_text(messages)
    data = json.loads(_clean_json(raw))
    return PlanResponse(schedule=data.get("schedule", []), summary=str(data.get("summary", "")), total_days=data.get("total_days", 0), daily_hours_recommended=data.get("daily_hours_recommended", 2.0))

@app.post("/lecture/chat", response_model=LectureChatResponse)
async def lecture_chat(req: LectureChatRequest):
    syllabus_str = ", ".join(req.full_syllabus)
    messages = [
        {"role": "system", "content": f"You are a tutor. Subject: {req.subject}. Syllabus: [{syllabus_str}]. If doubt asked, next_topic = {req.topic}. Output ONLY JSON: {{\"reply\": \"...\", \"next_topic\": \"...\", \"progress_pct\": 0.0}}"}
    ]
    for m in req.history[-6:]: messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})
    raw = await _chat_text(messages, temperature=0.7)
    data = json.loads(_clean_json(raw))
    return LectureChatResponse(reply=data.get("reply", ""), next_topic=data.get("next_topic", req.topic), progress_pct=data.get("progress_pct", 0.0))
