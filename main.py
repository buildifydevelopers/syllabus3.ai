# ============================================================
#  EduPlatform Backend — Hugging Face Meta LLaMA Version
#  Stack : FastAPI + HuggingFace Inference API (Meta LLaMA 3)
#  Model : meta-llama/Meta-Llama-3.1-8B-Instruct (free tier)
#          swap to Meta-Llama-3.1-70B-Instruct for better quality
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
    hf_api_key: str = ""                          # Hugging Face token (read)
    allowed_origin: str = "*"
    # Swap model here — no other code change needed
    hf_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

HF_API_URL = f"https://api-inference.huggingface.co/v1/chat/completions"


# ╔══════════════════════════════════════════════════════════╗
# ║  2. HF INFERENCE HELPERS                                 ║
# ╚══════════════════════════════════════════════════════════╝

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.hf_api_key}",
        "Content-Type": "application/json",
    }


async def _chat_text(
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Call HF chat completions, return plain text reply."""
    payload = {
        "model": settings.hf_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": max(temperature, 0.01),  # HF requires > 0
        "stream": False,
    }
    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(HF_API_URL, headers=_headers(), json=payload)

    if resp.status_code != 200:
        logger.error(f"HF API error {resp.status_code}: {resp.text}")
        raise HTTPException(status_code=502, detail=f"HF API error: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _system_teacher(subject: str, topic: str, mode: str = "text") -> dict:
    return {
        "role": "system",
        "content": f"""You are EduBot, an expert academic teacher specialising in {subject}.
Your ONLY job is to teach the topic: "{topic}".
RULES:
1. Stay on topic. If asked anything unrelated, redirect back to {topic}.
2. Never hallucinate. Only state facts you are certain about.
3. Teach one concept at a time. Confirm understanding before moving on.
4. Use correct academic terminology. Explain terms on first use.
5. {"Keep responses under 80 words. No markdown. Plain sentences only." if mode == "voice" else "Use markdown (headers, bold, bullets) to structure responses."}
6. End each reply with [TEACHING], [CHECK], or [RECAP].
""",
    }


def _infer_phase(message_count: int) -> str:
    if message_count <= 2:    return "INTRODUCTION"
    elif message_count <= 8:  return "CORE TEACHING"
    elif message_count <= 12: return "EXAMPLES & PRACTICE"
    else:                     return "RECAP & WRAP-UP"


# ╔══════════════════════════════════════════════════════════╗
# ║  3. SCHEMAS                                              ║
# ╚══════════════════════════════════════════════════════════╝

class Message(BaseModel):
    role: str
    content: str

class SyllabusRequest(BaseModel):
    subject: str
    raw_text: Optional[str] = None
    image_base64: Optional[str] = None   # ⚠️ LLaMA text-only — image not supported
    image_mime: Optional[str] = "image/jpeg"
    pdf_base64: Optional[str] = None     # Android extracts text → sends as raw_text

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
    missed_from_date: str
    exam_date: str
    daily_hours_recommended: float

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
# ║  4. APP + STARTUP                                        ║
# ╚══════════════════════════════════════════════════════════╝

app = FastAPI(title="EduPlatform AI — Meta LLaMA", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    if not settings.hf_api_key:
        raise RuntimeError("HF_API_KEY not set. Add Hugging Face token to .env")
    logger.info(f"Using model : {settings.hf_model}")
    logger.info(f"API URL     : {HF_API_URL}")


# ╔══════════════════════════════════════════════════════════╗
# ║  5. ROUTES                                               ║
# ╚══════════════════════════════════════════════════════════╝

@app.get("/health")
def health():
    return {"status": "healthy", "model": settings.hf_model}


# ── Syllabus Parser ───────────────────────────────────────
# ⚠️  LLaMA 3.1-8B is TEXT ONLY via HF Inference API
#    raw_text  → ✅ works
#    pdf_base64→ ✅ extract text on Android (PdfRenderer) → send as raw_text
#    image     → ❌ not supported — use Llama 3.2-Vision model OR OCR on device

@app.post("/syllabus/parse", response_model=SyllabusResponse)
async def parse_syllabus(req: SyllabusRequest):
    logger.info(f"/syllabus/parse subject={req.subject}")

    if req.raw_text:
        syllabus_content = req.raw_text
    elif req.pdf_base64:
        try:
            pdf_bytes = base64.b64decode(req.pdf_base64)
            syllabus_content = pdf_bytes.decode("utf-8", errors="ignore")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode PDF. Extract text on device and send as raw_text.")
    elif req.image_base64:
        raise HTTPException(
            status_code=400,
            detail="Image not supported with LLaMA text model. Use OCR on device and send extracted text as raw_text."
        )
    else:
        raise HTTPException(status_code=400, detail="Provide raw_text or pdf_base64.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic curriculum analyst. "
                "Extract topics from the provided syllabus. "
                "Output ONLY valid JSON — no markdown fences, no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"""Parse this syllabus for subject "{req.subject}".
Extract every distinct topic/chapter/unit.
Estimate realistic self-study time:
  foundational → 30-45 min | standard → 45-90 min | complex → 90-180 min

Output ONLY this JSON:
{{
  "topics": ["topic1", "topic2"],
  "estimated_total_hours": 42.5,
  "topic_details": [
    {{"topic": "topic1", "estimated_min": 60, "complexity": "standard"}}
  ]
}}

SYLLABUS:
{syllabus_content[:6000]}""",
        },
    ]

    try:
        raw = await _chat_text(messages, max_tokens=2048, temperature=0.01)
        logger.info(f"Raw LLaMA: {raw[:300]}")
        data = json.loads(_clean_json(raw))
        return SyllabusResponse(
            topics=data.get("topics", []),
            estimated_total_hours=data.get("estimated_total_hours", 0.0),
            topic_details=data.get("topic_details", []),
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON error: {e} | raw: {raw}")
        raise HTTPException(status_code=502, detail="LLaMA returned invalid JSON.")
    except Exception as e:
        logger.error(f"FAILED /syllabus/parse: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ── Study Plan ────────────────────────────────────────────

@app.post("/plan/generate", response_model=PlanResponse)
async def generate_plan(req: PlanRequest):
    logger.info(f"/plan/generate subject={req.subject}")

    exam_str = req.exam_date if req.exam_date else "Not specified"
    start = dt.strptime(req.start_date, "%Y-%m-%d")
    days = (dt.strptime(req.exam_date, "%Y-%m-%d") - start).days if req.exam_date else len(req.topics) * 2
    total_min = len(req.topics) * 60
    daily_hours = round(total_min / max(days, 1) / 60, 1)
    daily_hours = max(1.0, min(daily_hours, 8.0))
    duration_mins = int(daily_hours * 60)

    messages = [
        {
            "role": "system",
            "content": "You are a professional academic curriculum planner. Output ONLY valid JSON. No markdown fences.",
        },
        {
            "role": "user",
            "content": f"""Create a day-by-day study schedule.

Subject: {req.subject}
Topics: {json.dumps(req.topics)}
Start date: {req.start_date}
Exam date: {exam_str}
Days available: {days}
Daily duration: {duration_mins} minutes (FIXED)

Rules:
- Foundational = 1 day, complex = 2-3 days
- Prerequisites come first
- Revision day every 5 lecture days
- Last 3 days = revision/mock tests only
- No lectures on Sundays
- duration_mins = {duration_mins} for every entry

Output ONLY this JSON:
{{
  "schedule": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "topic": "topic name",
      "duration_mins": {duration_mins},
      "notes": "specific study tip",
      "type": "lecture"
    }}
  ],
  "summary": "3-sentence strategy overview",
  "total_days": {days},
  "daily_hours_recommended": {daily_hours}
}}
type values: "lecture" | "revision" | "exam" | "rest"
""",
        },
    ]

    try:
        raw = await _chat_text(messages, max_tokens=4096, temperature=0.3)
        logger.info(f"Raw plan: {raw[:300]}")
        data = json.loads(_clean_json(raw))
        return PlanResponse(
            schedule=data.get("schedule", []),
            summary=data.get("summary", ""),
            total_days=data.get("total_days", days),
            daily_hours_recommended=data.get("daily_hours_recommended", daily_hours),
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON error: {e}")
        raise HTTPException(status_code=502, detail="LLaMA returned invalid JSON.")
    except Exception as e:
        logger.error(f"FAILED /plan/generate: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ── Replan ────────────────────────────────────────────────

@app.post("/plan/replan", response_model=ReplanResponse)
async def replan(req: ReplanRequest):
    today_str = dt.now().strftime("%Y-%m-%d")
    days_left = (dt.strptime(req.exam_date, "%Y-%m-%d") - dt.now()).days
    total_min = len(req.remaining_topics) * 60
    new_daily_hours = round(total_min / max(days_left, 1) / 60, 1)
    new_daily_hours = max(1.0, min(new_daily_hours, 8.0))
    duration_mins = int(new_daily_hours * 60)

    messages = [
        {"role": "system", "content": "You are a recovery curriculum planner. Output ONLY valid JSON."},
        {
            "role": "user",
            "content": f"""Student missed study days. Create revised schedule.

Subject: {req.subject}
Remaining topics: {json.dumps(req.remaining_topics)}
Resume from: {today_str}
Exam date: {req.exam_date}
Days left: {days_left}
Daily duration: {duration_mins} minutes

Output ONLY this JSON:
{{
  "schedule": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "topic": "topic name",
      "duration_mins": {duration_mins},
      "notes": "study tip",
      "type": "lecture"
    }}
  ],
  "summary": "honest 3-sentence recovery strategy"
}}
""",
        },
    ]

    try:
        raw = await _chat_text(messages, max_tokens=4096, temperature=0.3)
        data = json.loads(_clean_json(raw))
        return ReplanResponse(
            schedule=data.get("schedule", []),
            summary=data.get("summary", ""),
            daily_hours_recommended=new_daily_hours,
        )
    except Exception as e:
        logger.error(f"FAILED /plan/replan: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ── Lecture Intro ─────────────────────────────────────────

@app.post("/lecture/intro", response_model=LectureIntroResponse)
async def lecture_intro(req: LectureIntroRequest):
    messages = [
        _system_teacher(req.subject, req.topic),
        {
            "role": "user",
            "content": f"""Write Phase 1 (INTRODUCTION) for lecture on "{req.topic}".
Include:
1. Welcome sentence naming topic.
2. Definition (2-3 sentences, correct terminology).
3. Why it matters — one real-world use.
4. Prerequisites (or "No prior knowledge needed").
5. Roadmap (numbered list of what will be covered).
6. "Type 'ready' or ask any question to begin! 🎓"

Markdown format. 200-280 words. Do NOT teach content yet.
[TEACHING]""",
        },
    ]
    try:
        reply = await _chat_text(messages, max_tokens=512, temperature=0.01)
        return LectureIntroResponse(intro=reply)
    except Exception as e:
        logger.error(f"FAILED /lecture/intro: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ── Lecture Chat (SSE streaming) ──────────────────────────

@app.post("/lecture/chat")
async def lecture_chat(req: LectureChatRequest):
    logger.info(f"/lecture/chat topic={req.topic}")

    message_count = len(req.history)
    current_phase = _infer_phase(message_count)
    user_msg_count = len([m for m in req.history if m.role == "user"])
    progress = min(95.0, (user_msg_count / 20) * 100)

    # Keep first 2 + last 10 to preserve intro context
    history_slice = (req.history[:2] + req.history[-10:]) if len(req.history) > 12 else req.history

    messages = [_system_teacher(req.subject, req.topic, req.mode)]
    for m in history_slice:
        messages.append({
            "role": "user" if m.role == "user" else "assistant",
            "content": m.content,
        })
    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": settings.hf_model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.01,
        "stream": True,
    }

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream("POST", HF_API_URL, headers=_headers(), json=payload) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk_str = line[6:].strip()
                            if chunk_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(chunk_str)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield f"data: {json.dumps({'reply': delta, 'phase': current_phase, 'progress_pct': progress})}\n\n"
                            except Exception:
                                continue
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'reply': f'Error: {str(e)}', 'phase': 'ERROR', 'progress_pct': 0.0})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Doubt Solver ──────────────────────────────────────────

@app.post("/doubt/ask", response_model=DoubtResponse)
async def ask_doubt(req: DoubtRequest):
    ctx_str = "\n".join(
        f"{'TEACHER' if m.role == 'assistant' else 'STUDENT'}: {m.content[:300]}"
        for m in (req.context or [])[-4:]
    ) or "None"

    messages = [
        _system_teacher(req.subject, req.topic),
        {
            "role": "user",
            "content": f"""Answer student doubt about "{req.topic}".

Recent context:
{ctx_str}

Doubt: "{req.question}"

Structure:
**Direct Answer** (1-2 sentences)
**Explanation** (3-5 sentences)
**Example** (1 concrete example)
**Common Mistake**
**In short:** (one-line summary)

End: "Does this clear your doubt? Feel free to ask a follow-up! 💡"
""",
        },
    ]
    try:
        answer = await _chat_text(messages, max_tokens=512, temperature=0.01)
        return DoubtResponse(answer=answer)
    except Exception as e:
        logger.error(f"FAILED /doubt/ask: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ── Session Summary ───────────────────────────────────────

@app.post("/lecture/summary", response_model=SummaryResponse)
async def lecture_summary(req: SummaryRequest):
    conversation = "\n".join(
        f"{'TEACHER' if m.role == 'assistant' else 'STUDENT'}: {m.content}"
        for m in req.history
    )[:4000]

    messages = [
        {"role": "system", "content": "Generate exam-ready study notes from a lecture transcript. Use markdown."},
        {
            "role": "user",
            "content": f"""Subject: {req.subject} | Topic: {req.topic}

Transcript:
{conversation}

Format:
## {req.topic} — Lecture Summary
### Key Concepts Covered
### Detailed Notes
### Examples Discussed
### Important Definitions
### Common Mistakes to Avoid
### What to Study Next

Only include what was discussed. Under 400 words.
""",
        },
    ]
    try:
        summary = await _chat_text(messages, max_tokens=800, temperature=0.01)
        return SummaryResponse(summary=summary)
    except Exception as e:
        logger.error(f"FAILED /lecture/summary: {e}")
        raise HTTPException(status_code=502, detail=str(e))
