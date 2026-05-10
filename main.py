# ============================================================
#  EduPlatform Backend — stateless AI service
#  Stack : FastAPI + NVIDIA NIM (OpenAI-compatible)
#  Design: Frontend owns ALL data/storage.
#          This backend receives context, calls NIM,
#          and returns the AI response. Nothing is stored here.
# ============================================================

import json
import re
from typing import List, Optional

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# ╔══════════════════════════════════════════════════════════╗
# ║  1. CONFIG                                               ║
# ╚══════════════════════════════════════════════════════════╝

class Settings(BaseSettings):
    nvidia_api_key: str = ""
    nim_model: str = "meta/llama-3.3-70b-instruct"   # swap any NIM model here
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


# ╔══════════════════════════════════════════════════════════╗
# ║  2. NIM HELPERS                                          ║
# ╚══════════════════════════════════════════════════════════╝

def _client() -> OpenAI:
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nim_base_url,
    )


def _chat(
    system: str,
    history: List[dict],        # [{"role": "user"|"assistant", "content": str}, ...]
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Single helper for all NIM calls.
    Builds: system → history → new user message → returns reply text.
    """
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = _client().chat.completions.create(
        model=settings.nim_model,
        messages=messages,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _teacher_persona(subject: str, topic: str, mode: str = "text") -> str:
    """
    Master behavioral contract injected into every lecture prompt.
    """
    return f"""
=== AI TEACHER IDENTITY & STRICT RULES ===

You are EduBot, an expert academic teacher specialising in {subject}.
Your ONLY job right now is to teach the topic: "{topic}".

ABSOLUTE RULES — follow these without exception:

1. STAY ON TOPIC: Every response must be directly about "{topic}" in {subject}.
   If the student asks about anything unrelated, say:
   "That's outside today's topic. Let's stay focused on {topic}."
   Then immediately return to teaching. Never answer off-topic questions.

2. NEVER HALLUCINATE: Only state facts you are certain about.
   If unsure, say "Let's reason through this carefully together."
   Never invent formulas, dates, names, or definitions.

3. STRUCTURED TEACHING — follow this flow every session:
   Phase 1 — INTRODUCTION  : Define the topic, why it matters, real-world use.
   Phase 2 — CORE TEACHING : Explain concepts one at a time.
   Phase 3 — EXAMPLES      : Give 1-2 concrete examples per concept.
   Phase 4 — CHECK         : Ask the student a question to verify understanding.
   Phase 5 — RECAP         : Summarise key points before ending.

4. ONE CONCEPT AT A TIME: Teach one idea, confirm understanding, then move on.

5. ACADEMIC LANGUAGE: Use correct subject-specific terminology.
   Always explain a term the first time you use it.

6. NO CASUAL CHAT: Do not discuss anything outside {subject} / {topic}.

7. ENCOURAGE HONESTLY: Say "Good thinking!" only when the student is correct.

8. {"VOICE MODE — keep each response under 80 words. Short, clear sentences only. No markdown." if mode == "voice" else "TEXT MODE — use markdown (headers, bold, bullets) to structure your response."}

9. End each response with one of:
   [TEACHING] — still explaining a concept
   [CHECK]    — just asked a comprehension question
   [RECAP]    — summarising at the end

=== END OF RULES ===
"""


def _infer_phase(message_count: int) -> str:
    if message_count <= 2:    return "INTRODUCTION"
    elif message_count <= 8:  return "CORE TEACHING"
    elif message_count <= 12: return "EXAMPLES & PRACTICE"
    else:                     return "RECAP & WRAP-UP"


# ╔══════════════════════════════════════════════════════════╗
# ║  3. REQUEST / RESPONSE SCHEMAS                           ║
# ╚══════════════════════════════════════════════════════════╝

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


# ── /plan/generate ────────────────────────────────────────
class PlanRequest(BaseModel):
    subject: str
    topics: List[str]
    start_date: str
    exam_date: Optional[str] = None
    daily_hours: int = 2

class PlanResponse(BaseModel):
    schedule: list
    summary: str
    total_days: int


# ── /lecture/intro ────────────────────────────────────────
class LectureIntroRequest(BaseModel):
    subject: str
    topic: str

class LectureIntroResponse(BaseModel):
    intro: str


# ── /lecture/chat ─────────────────────────────────────────
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


# ── /doubt/ask ────────────────────────────────────────────
class DoubtRequest(BaseModel):
    subject: str
    topic: str
    question: str
    context: Optional[List[Message]] = []

class DoubtResponse(BaseModel):
    answer: str


# ── /lecture/summary ─────────────────────────────────────
class SummaryRequest(BaseModel):
    subject: str
    topic: str
    history: List[Message]

class SummaryResponse(BaseModel):
    summary: str


# ╔══════════════════════════════════════════════════════════╗
# ║  4. APP                                                  ║
# ╚══════════════════════════════════════════════════════════╝

app = FastAPI(
    title="EduPlatform AI Service",
    description=(
        "Stateless AI backend. Frontend sends all context in the request body. "
        "Backend calls NVIDIA NIM and returns the AI response. No data is stored here."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ╔══════════════════════════════════════════════════════════╗
# ║  5. ROUTES                                               ║
# ╚══════════════════════════════════════════════════════════╝

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "EduPlatform AI", "version": "2.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ── Study Plan ────────────────────────────────────────────

@app.post("/plan/generate", response_model=PlanResponse, tags=["Study Plan"])
def generate_plan(req: PlanRequest):
    from datetime import datetime as dt
    start = dt.strptime(req.start_date, "%Y-%m-%d")
    exam  = dt.strptime(req.exam_date, "%Y-%m-%d") if req.exam_date else None
    days  = (exam - start).days if exam else len(req.topics) * 2

    prompt = f"""You are a professional academic curriculum planner.

TASK: Create a complete, realistic day-by-day study schedule.

INPUT:
  Subject        : {req.subject}
  Topics         : {json.dumps(req.topics)}
  Start date     : {req.start_date}
  Exam date      : {req.exam_date or 'Not specified'}
  Days available : {days}
  Daily hours    : {req.daily_hours}

PLANNING RULES:
  - Assess each topic's complexity (foundational = 1 day, complex = 2-3 days)
  - Prerequisites of other topics must come first
  - Insert a "Revision" day after every 5 lecture days
  - Last 3 days before exam = revision/mock tests only
  - Do NOT assign lectures on Sundays (rest day)
  - duration_mins = daily_hours * 60 (max 180)
  - notes must be a specific actionable study tip for that topic

Respond ONLY with valid JSON, no markdown fences:
{{
  "schedule": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "topic": "exact topic name",
      "duration_mins": 120,
      "notes": "specific study tip",
      "type": "lecture"
    }}
  ],
  "summary": "3-sentence strategy overview",
  "total_days": {days}
}}
type values: "lecture" | "revision" | "exam" | "rest"
"""
    try:
        text = _chat(system="You are a professional academic curriculum planner.", history=[], user_message=prompt, temperature=0.3)
        data = json.loads(_clean_json(text))
        return PlanResponse(
            schedule=data.get("schedule", []),
            summary=data.get("summary", ""),
            total_days=data.get("total_days", days),
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"NIM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")


# ── Lecture Intro ─────────────────────────────────────────

@app.post("/lecture/intro", response_model=LectureIntroResponse, tags=["Lecture"])
def lecture_intro(req: LectureIntroRequest):
    user_prompt = f"""
TASK: Write the Phase 1 (INTRODUCTION) opening for today's lecture on "{req.topic}".

Your introduction MUST include ALL of the following in this order:
1. One-sentence welcome that names the topic explicitly.
2. WHAT: Precise definition of "{req.topic}" in 2-3 sentences using correct terminology.
3. WHY: Why this topic matters — one real-world application or use case.
4. PREREQUISITES: 1-2 concepts the student should already know.
   (If foundational, say "No prior knowledge needed.")
5. TODAY'S ROADMAP: Numbered list of exactly what will be covered.
6. Closing line: "Type 'ready' or ask any clarifying question to begin! 🎓"

FORMAT: Use markdown. Bold key terms on first use.
LENGTH: 200-280 words.
Do NOT teach the content yet — this is the introduction only.

[TEACHING]
"""
    try:
        reply = _chat(
            system=_teacher_persona(req.subject, req.topic),
            history=[],
            user_message=user_prompt,
            temperature=0.0,
        )
        return LectureIntroResponse(intro=reply)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")


# ── Lecture Chat ──────────────────────────────────────────

@app.post("/lecture/chat", response_model=LectureChatResponse, tags=["Lecture"])
def lecture_chat(req: LectureChatRequest):
    message_count = len(req.history)
    current_phase = _infer_phase(message_count)

    phase_instruction = {
        "INTRODUCTION": """
You are in Phase 1 (INTRODUCTION).
If the student says "ready" or similar → begin Phase 2 immediately.
Otherwise answer their clarifying question briefly, then re-invite them to start.
""",
        "CORE TEACHING": f"""
You are in Phase 2 (CORE TEACHING).
- Teach ONE sub-concept of "{req.topic}" at a time.
- State and bold the concept name.
- Explain in 3-5 sentences with correct terminology.
- Give ONE real-world example.
- End by asking a comprehension question.
- Do NOT move to the next concept until the student confirms understanding.
""",
        "EXAMPLES & PRACTICE": f"""
You are in Phase 3 (EXAMPLES & PRACTICE).
- Give a concrete worked example or problem about "{req.topic}".
- Walk through it step by step.
- Ask the student to solve a similar problem.
- If they answer: correct → explain why; wrong → explain the mistake clearly.
""",
        "RECAP & WRAP-UP": """
You are in Phase 5 (RECAP).
- Summarise all key concepts as a numbered list.
- Highlight the 2-3 most important takeaways.
- Suggest what to review next.
- End with: "Great work today! Type /end to finish the lecture. 🎓"
""",
    }.get(current_phase, "")

    off_topic_guard = f"""
IMPORTANT — Before responding, check: Is the student message related to "{req.topic}" in {req.subject}?
- YES → teach normally per phase instructions.
- NO  → reply: "That's outside today's scope! Let's stay on **{req.topic}**. [one-line redirect] [TEACHING]"
  Do NOT answer the off-topic question.
"""

    system = (
        _teacher_persona(req.subject, req.topic, req.mode)
        + f"\nCURRENT PHASE: {current_phase} (message {message_count + 1} of session)\n"
        + phase_instruction
        + off_topic_guard
    )

    # Convert history to OpenAI message format (last 12 turns)
    nim_history = [
        {"role": m.role if m.role == "user" else "assistant", "content": m.content}
        for m in req.history[-12:]
    ]

    try:
        reply = _chat(
            system=system,
            history=nim_history,
            user_message=req.message,
            temperature=0.0,
        )
        progress = min(95.0, len([m for m in req.history if m.role == "user"]) * 10.0)
        return LectureChatResponse(reply=reply, phase=current_phase, progress_pct=progress)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")


# ── Doubt Solver ──────────────────────────────────────────

@app.post("/doubt/ask", response_model=DoubtResponse, tags=["Doubt Solver"])
def ask_doubt(req: DoubtRequest):
    ctx_str = ""
    if req.context:
        ctx_str = "\n".join(
            f"{'TEACHER' if m.role == 'assistant' else 'STUDENT'}: {m.content[:300]}"
            for m in req.context[-4:]
        )

    user_prompt = f"""
TASK: Answer the following student doubt.

Recent lecture context:
{ctx_str or "No prior context."}

Student's doubt: "{req.question}"

DOUBT RESOLUTION RULES:
1. RELEVANCE CHECK: If the doubt is NOT about "{req.topic}" in {req.subject}, respond:
   "This doubt is outside today's topic ({req.topic}). Please ask after the lecture."
   Do NOT answer unrelated doubts.

2. If relevant, answer using EXACTLY this structure:

   **Direct Answer** (1-2 sentences — state the answer immediately)

   **Explanation** (3-5 sentences — explain clearly with correct terminology)

   **Example** (1 concrete example)

   **Common Mistake** (what students often get wrong about this)

   **In short:** (one-line summary to memorise)

3. Only state facts you are certain about.
4. Use markdown formatting.
5. End with: "Does this clear your doubt? Feel free to ask a follow-up! 💡"
"""
    try:
        answer = _chat(
            system=_teacher_persona(req.subject, req.topic),
            history=[],
            user_message=user_prompt,
            temperature=0.0,
        )
        return DoubtResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")


# ── Session Summary ───────────────────────────────────────

@app.post("/lecture/summary", response_model=SummaryResponse, tags=["Lecture"])
def lecture_summary(req: SummaryRequest):
    conversation = "\n".join(
        f"{'TEACHER' if m.role == 'assistant' else 'STUDENT'}: {m.content}"
        for m in req.history
    )[:4000]

    user_prompt = f"""
Subject : {req.subject}
Topic   : {req.topic}

Lecture transcript:
{conversation}

Generate a structured summary using EXACTLY this format:

## {req.topic} — Lecture Summary

### Key Concepts Covered
(Bullet list of every concept taught, one line each)

### Detailed Notes
(For each concept: 2-3 sentence explanation with correct terminology)

### Examples Discussed
(Any examples, analogies, or problems from the lecture)

### Important Definitions
(Technical terms defined — format: **Term**: definition)

### Common Mistakes to Avoid
(Mistakes or misconceptions highlighted during the lecture)

### What to Study Next
(1-2 logical follow-up topics)

RULES:
- Only include content actually discussed in the transcript.
- Do not add external information.
- Use academic language appropriate for {req.subject}.
- Keep total under 400 words.
"""
    try:
        summary = _chat(
            system="You are generating exam-ready academic study notes from a completed lecture.",
            history=[],
            user_message=user_prompt,
            temperature=0.0,
        )
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")
