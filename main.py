# ============================================================
#  EduPlatform Backend — NVIDIA NIM Version (Optimized)
#  Stack : FastAPI + NVIDIA NIM API (OpenAI-compatible)
#  Model : meta/llama-3.1-8b-instruct (Llama 3.1)
# ============================================================

import base64
import json
import re
import logging
from datetime import datetime as dt
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EduPlatform")

# ╔══════════════════════════════════════════════════════════╗
# ║  1. CONFIG                                               ║
# ╚══════════════════════════════════════════════════════════╝

class Settings(BaseSettings):
    nvidia_api_key: str = ""
    allowed_origin: str = "*"
    nvidia_model: str = "meta/llama-3.1-8b-instruct"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ╔══════════════════════════════════════════════════════════╗
# ║  2. HELPERS                                              ║
# ╚══════════════════════════════════════════════════════════╝

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }

async def _chat_text(messages: list, max_tokens: int = 1024, temperature: float = 0.2) -> str:
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
        logger.error(f"NVIDIA API error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=502, detail=f"AI Engine error: {resp.status_code}")

    return resp.json()["choices"][0]["message"]["content"].strip()

def _clean_json(text: str) -> str:
    """Robustly extract JSON and handle common AI syntax errors."""
    text = text.strip()
    # Remove markdown code blocks if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    
    # Try to find the outermost curly braces
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    
    # Fix trailing commas in lists/objects which break json.loads
    text = re.sub(r",\s*([\]\}])", r"\1", text)
    return text

def _system_teacher(subject: str, topic: str, mode: str = "text") -> dict:
    return {
        "role": "system",
        "content": f"""You are EduBot, an expert academic teacher specializing in {subject}.
Your goal is to teach the topic: "{topic}".
RULES:
1. Stay strictly on topic.
2. Teach one concept at a time.
3. Use markdown for structure (bold, bullets).
4. End replies with [TEACHING], [CHECK], or [RECAP].
""",
    }

def _infer_phase(message_count: int) -> str:
    if message_count <= 2:    return "INTRODUCTION"
    elif message_count <= 8:  return "CORE TEACHING"
    else:                     return "RECAP"

# ╔══════════════════════════════════════════════════════════╗
# ║  3. SCHEMAS                                              ║
# ╚══════════════════════════════════════════════════════════╝

class Message(BaseModel):
    role: str
    content: str

class SyllabusRequest(BaseModel):
    subject: str
    raw_text: Optional[str] = None
    image_base64: Optional[str] = None
    pdf_base64: Optional[str] = None

class SyllabusResponse(BaseModel):
    topics: List[str]
    estimated_total_hours: float
    topic_details: list

class PlanRequest(BaseModel):
    subject: str
    topics: List[str]
    start_date: str
    exam_date: Optional[str] = None

class PlanResponse(BaseModel):
    schedule: list
    summary: str
    total_days: int
    daily_hours_recommended: float

class ReplanRequest(BaseModel):
    subject: str
    remaining_topics: List[str]
    exam_date: str

class ReplanResponse(BaseModel):
    schedule: list
    summary: str
    daily_hours_recommended: float

class LectureIntroRequest(BaseModel):
    subject: str
    topic: str

class LectureIntroResponse(BaseModel):
    intro: str

class LectureChatRequest(BaseModel):
    subject: str
    topic: str
    history: List[Message]
    message: str
    mode: str = "text"

class LectureChatResponse(BaseModel):
    reply: str
    phase: str
    progress_pct: float

class SummaryRequest(BaseModel):
    subject: str
    topic: str
    history: List[Message]

class SummaryResponse(BaseModel):
    summary: str

class DoubtRequest(BaseModel):
    subject: str
    topic: str
    question: str
    context: Optional[List[Message]] = []

class DoubtResponse(BaseModel):
    answer: str

# ╔══════════════════════════════════════════════════════════╗
# ║  4. APP                                                  ║
# ╚══════════════════════════════════════════════════════════╝

app = FastAPI(title="EduPlatform AI — NVIDIA NIM", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    if not settings.nvidia_api_key:
        logger.warning("NVIDIA_API_KEY is not set.")

@app.get("/health")
def health():
    return {"status": "healthy", "model": settings.nvidia_model}

# ╔══════════════════════════════════════════════════════════╗
# ║  5. ROUTES                                               ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/syllabus/parse", response_model=SyllabusResponse)
async def parse_syllabus(req: SyllabusRequest):
    content = req.raw_text or "No content provided."
    logger.info(f"/syllabus/parse subject={req.subject}")

    messages = [
        {"role": "system", "content": "You are an academic curriculum analyst. Output ONLY valid JSON. No conversational text."},
        {"role": "user", "content": f"""Parse this syllabus for {req.subject}. 
Output JSON with:
- "topics": list of strings
- "estimated_total_hours": float
- "topic_details": list of objects {{"topic": string, "estimated_min": int, "complexity": string}}

SYLLABUS: 
{content[:5000]}"""}
    ]
    raw = ""
    try:
        # Use lower temperature for structured data tasks
        raw = await _chat_text(messages, temperature=0.1, max_tokens=2500)
        cleaned = _clean_json(raw)
        data = json.loads(cleaned)
        return SyllabusResponse(**data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Failure: {e} | Processed string: {cleaned if 'cleaned' in locals() else 'N/A'}")
        logger.error(f"Raw AI Output: {raw}")
        raise HTTPException(status_code=502, detail="AI produced invalid JSON formatting.")
    except Exception as e:
        logger.error(f"Syllabus parse error: {e}")
        raise HTTPException(status_code=502, detail="AI parsing failed.")

@app.post("/plan/generate", response_model=PlanResponse)
async def generate_plan(req: PlanRequest):
    logger.info(f"/plan/generate subject={req.subject}")
    messages = [
        {"role": "system", "content": "You are a rigid study planner. You ONLY output valid JSON. You NEVER change the structure."},
        {"role": "user", "content": f"""Create a study plan for {req.subject}. 
Topics: {req.topics}
Start Date: {req.start_date}

Output EXACTLY this JSON structure:
{{
  "schedule": [
    {{
      "day": 1,
      "date": "2026-05-06",
      "topic": "Topic Name",
      "duration_mins": 60,
      "notes": "Tip",
      "type": "lecture"
    }}
  ],
  "summary": "Plain text summary string here",
  "total_days": 10,
  "daily_hours_recommended": 2.0
}}

CRITICAL: 
- 'schedule' MUST be a LIST (array), not a dictionary. 
- 'summary' MUST be a STRING, not an object.
"""}
    ]
    raw = ""
    try:
        raw = await _chat_text(messages, max_tokens=3000, temperature=0.1)
        cleaned = _clean_json(raw)
        data = json.loads(cleaned)
        
        # RESILIENCE: If AI returns schedule as a dict, convert to list
        schedule_data = data.get("schedule", [])
        if isinstance(schedule_data, dict):
            logger.warning("AI returned dictionary schedule. Converting to list.")
            new_list = []
            for date_key, details in schedule_data.items():
                if isinstance(details, dict):
                    details["date"] = date_key
                    new_list.append(details)
            schedule_data = new_list

        # RESILIENCE: If summary is an object/dict, convert to string
        summary_data = data.get("summary", "")
        if not isinstance(summary_data, str):
            summary_data = json.dumps(summary_data)

        return PlanResponse(
            schedule=schedule_data,
            summary=summary_data,
            total_days=data.get("total_days", 0),
            daily_hours_recommended=data.get("daily_hours_recommended", 2.0)
        )
    except Exception as e:
        logger.error(f"Plan Generation Failed: {e} | Raw: {raw}")
        raise HTTPException(status_code=502, detail="AI failed to generate a valid study plan structure.")

@app.post("/lecture/intro", response_model=LectureIntroResponse)
async def lecture_intro(req: LectureIntroRequest):
    messages = [
        _system_teacher(req.subject, req.topic),
        {"role": "user", "content": f"Give a 200-word introduction to {req.topic}. Explain why it's important and what we will cover."}
    ]
    try:
        reply = await _chat_text(messages)
        return LectureIntroResponse(intro=reply)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/lecture/chat")
async def lecture_chat(req: LectureChatRequest):
    message_count = len(req.history)
    phase = _infer_phase(message_count)
    
    # History Slicing for token efficiency
    history_slice = req.history[-8:] if len(req.history) > 8 else req.history
    
    messages = [_system_teacher(req.subject, req.topic, req.mode)]
    for m in history_slice:
        messages.append({"role": m.role if m.role in ["user", "assistant"] else "user", "content": m.content})
    messages.append({"role": "user", "content": req.message})

    async def generate():
        payload = {
            "model": settings.nvidia_model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream("POST", NVIDIA_API_URL, headers=_headers(), json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk_str = line[6:].strip()
                            if chunk_str == "[DONE]": break
                            try:
                                chunk = json.loads(chunk_str)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield f"data: {json.dumps({'reply': delta, 'phase': phase, 'progress_pct': 50.0})}\n\n"
                            except: continue
        except Exception as e:
            yield f"data: {json.dumps({'reply': 'Error: ' + str(e), 'phase': 'ERROR', 'progress_pct': 0.0})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/doubt/ask", response_model=DoubtResponse)
async def ask_doubt(req: DoubtRequest):
    ctx_str = "\n".join([f"{m.role}: {m.content[:200]}" for m in (req.context or [])[-3:]])
    messages = [
        _system_teacher(req.subject, req.topic),
        {"role": "user", "content": f"Context:\n{ctx_str}\n\nQuestion: {req.question}"}
    ]
    try:
        answer = await _chat_text(messages)
        return DoubtResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/lecture/summary", response_model=SummaryResponse)
async def lecture_summary(req: SummaryRequest):
    conv = "\n".join([f"{m.role}: {m.content}" for m in req.history])[:3000]
    messages = [
        {"role": "system", "content": "Summarize the following study session into exam notes."},
        {"role": "user", "content": f"Subject: {req.subject}. Topic: {req.topic}. Conversation:\n{conv}"}
    ]
    try:
        summary = await _chat_text(messages, max_tokens=1024)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
