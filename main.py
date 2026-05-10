# ============================================================
#  EduPlatform Backend — stateless AI service
#  Stack : FastAPI + NVIDIA NIM (OpenAI-compatible)
#  Design: Frontend owns ALL data/storage.
#          This backend receives context, calls NIM,
#          and returns the AI response. Nothing is stored here.
# ============================================================

import json
import re
import logging
import traceback
import httpx
from typing import List, Optional

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EduPlatform")


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
        http_client=httpx.Client(),
    )


def _chat(
    system: str,
    history: List[dict],        # [{"role": "user"|"assistant", "content": str}, ...]
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Single helper for all NIM calls.
    Builds: system → history → new user message → returns reply text.
    """
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = _client().chat.completions.create(
            model=settings.nim_model,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("NIM Chat Error: %s", e)
        logger.error(traceback.format_exc())
        raise e


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _teacher_persona(subject: str, topic: str, type: str = "lecture", mode: str = "text") -> str:
    """
    Master behavioral contract injected into every lecture prompt.
    Adapts based on whether this is a Lecture, Test, or Revision.
    """
    
    base_instructions = ""
    if type == "test" or type == "mock_exam":
        base_instructions = f"""
You are now an EXAMINER. The session is a {type.upper()} on "{topic}".
1. DO NOT TEACH. Your goal is to assess knowledge.
2. Ask 3-5 structured questions one by one.
3. Wait for the student's answer before asking the next question.
4. After all questions, provide a Score (0-10) and detailed Feedback.
5. End by saying "[TOPIC_COMPLETED]" to move to the next session.
"""
    elif type == "revision":
        base_instructions = f"""
You are now a REVISION COACH. The topic is "{topic}".
1. Briefly summarize the 3 most critical points of "{topic}".
2. Ask the student if they have any specific doubts or parts they find difficult.
3. Conduct a quick rapid-fire quiz (2 questions).
"""
    else:
        # Default Lecture Persona
        base_instructions = f"""
You are EduBot, an expert academic teacher specialising in {subject}.
Your ONLY job right now is to teach the topic: "{topic}".

Phase 1 — INTRODUCTION  : Define the topic, why it matters, real-world use.
Phase 2 — CORE TEACHING : Explain concepts concisely with bold names and examples.
Phase 3 — CHECK         : Ask the student a question to verify understanding.
Phase 4 — RECAP         : Summarise key points.
IF THE STUDENT SAYS 'NEXT' OR 'DONE', IMMEDIATELY FINISH THIS TOPIC.
"""

    return f"""
=== AI TEACHER IDENTITY & STRICT RULES ===
{base_instructions}

ABSOLUTE RULES:
- STAY ON TOPIC: Every response must be directly about "{topic}".
- NO CASUAL CHAT: Do not discuss anything outside {subject} / {topic}.
- {"VOICE MODE — keep each response under 80 words. Short, clear sentences only. No markdown." if mode == "voice" else "TEXT MODE — use markdown (headers, bold, bullets) to structure your response."}
- End each response with [TEACHING], [CHECK], or [RECAP].
- When the topic is fully covered or the test is done, end with [TOPIC_COMPLETED].
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


# ── /syllabus/parse ───────────────────────────────────────
class SyllabusParseRequest(BaseModel):
    raw_text: str                        # pasted syllabus text from frontend
    subject: Optional[str] = None       # hint; AI infers if omitted

class SyllabusTopic(BaseModel):
    unit: str
    topic: str
    subtopics: List[str]
    estimated_days: int
    difficulty: str                      # "easy" | "medium" | "hard"

class SyllabusParseResponse(BaseModel):
    subject: str
    topics: List[SyllabusTopic]
    total_topics: int
    recommended_order: List[str]         # topic names in suggested study order


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

    prompt = f"""You are a professional Academic Tuition Planner.

TASK: Create a comprehensive day-by-day study and assessment schedule for the student.

INPUT:
  Subject        : {req.subject}
  Topics         : {json.dumps(req.topics)}
  Start Date     : {req.start_date}
  Exam Date      : {req.exam_date or 'Not specified'}
  Daily Hours    : {req.daily_hours}
  Days Available : {days}

PLANNING RULES:
  - Sequence topics logically by difficulty and prerequisites.
  - For every 6 days of learning, add 1 "Weekly Test" day.
  - For every 4 weeks of learning, add 1 "Monthly Grand Test" day.
  - The last 7 days before the Exam Date must be "Final Mock Exams" and "Intensive Revision".
  - Each entry must specify:
    - topic: The name of the topic or "Weekly Test" / "Monthly Test".
    - type: "lecture" | "test" | "revision" | "mock_exam"
    - notes: Specific study tip or test focus areas.
    - duration_mins: daily_hours * 60
    - day: sequential day number
    - date: YYYY-MM-DD (calculate based on start_date)

Respond ONLY with valid JSON, no markdown fences:
{{
  "schedule": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "topic": "Topic Name",
      "duration_mins": 120,
      "notes": "Focus on...",
      "type": "lecture"
    }}
  ],
  "summary": "Your personalized 1-to-1 tuition strategy."
}}
"""
    try:
        text = _chat(system="You are an expert AI Tuition Master.", history=[], user_message=prompt, temperature=0.3)
        data = json.loads(_clean_json(text))
        
        schedule = data.get("schedule", [])
        return PlanResponse(
            schedule=schedule,
            summary=data.get("summary", ""),
            total_days=len(schedule),
        )
    except json.JSONDecodeError as e:
        logger.error("JSON Parse Error: %s | text: %s", e, text[:500] if 'text' in locals() else "N/A")
        raise HTTPException(status_code=502, detail=f"NIM returned invalid JSON: {e}")
    except Exception as e:
        logger.error("Syllabus Parse Error: %s", e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")


# ── Lecture Intro ─────────────────────────────────────────

@app.post("/lecture/intro", response_model=LectureIntroResponse, tags=["Lecture"])
def lecture_intro(req: LectureIntroRequest):
    if req.type == "test" or req.type == "mock_exam":
        user_prompt = f"""
TASK: Open the {req.type.upper()} for "{req.topic}".
1. Welcome the student to the assessment.
2. Briefly explain the test format (3-5 questions).
3. Wish them luck and ask if they are ready to begin.
"""
    elif req.type == "revision":
        user_prompt = f"""
TASK: Open the REVISION session for "{req.topic}".
1. Welcome the student back.
2. State why revision of "{req.topic}" is important.
3. Ask if they have any initial questions or if they are ready for the summary.
"""
    else:
        user_prompt = f"""
TASK: Write the Phase 1 (INTRODUCTION) opening for today's lecture on "{req.topic}".
Your introduction MUST include: Welcome, Definition of "{req.topic}", Why it matters, and today's Roadmap.
Closing line: "Type 'ready' or ask any clarifying question to begin! 🎓"
"""
    try:
        reply = _chat(
            system=_teacher_persona(req.subject, req.topic, req.type),
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
    if req.type == "test" or req.type == "mock_exam":
        system = _teacher_persona(req.subject, req.topic, req.type, req.mode)
        current_phase = "TESTING"
    elif req.type == "revision":
        system = _teacher_persona(req.subject, req.topic, req.type, req.mode)
        current_phase = "REVISION"
    else:
        message_count = len(req.history)
        current_phase = _infer_phase(message_count)
        phase_instruction = {
            "INTRODUCTION": "If student says ready -> start core teaching.",
            "CORE TEACHING": "Teach concepts concisely with bold names and examples. Ask if ready for next.",
            "EXAMPLES & PRACTICE": "Give a problem, wait for answer, provide feedback.",
            "RECAP & WRAP-UP": "Summarize key points. End with [TOPIC_COMPLETED].",
        }.get(current_phase, "")
        
        system = (
            _teacher_persona(req.subject, req.topic, req.type, req.mode)
            + f"\nCURRENT PHASE: {current_phase}\n"
            + phase_instruction
        )

    # Convert history to OpenAI message format
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


# ── Syllabus Parser ───────────────────────────────────────

@app.post("/syllabus/parse", response_model=SyllabusParseResponse, tags=["Syllabus"])
def parse_syllabus(req: SyllabusParseRequest):
    """
    Frontend sends: raw syllabus text (pasted from PDF/doc), optional subject hint.
    Returns: structured topic list with difficulty, estimated days, recommended order.
    Frontend uses topics[] to seed /plan/generate and the topic selector.
    """
    prompt = f"""You are an expert academic curriculum analyst.

TASK: Parse the raw syllabus text below into a structured JSON topic list.

Subject hint: {req.subject or "Infer from content"}

Raw syllabus:
\"\"\"
{req.raw_text[:6000]}
\"\"\"

RULES:
1. Infer subject name if not provided.
2. Group content into logical units/chapters.
3. Estimate difficulty per topic: "easy" | "medium" | "hard"
   - easy   : definitional, recall-based
   - medium : requires understanding + application
   - hard   : multi-step reasoning, derivations, or complex problem-solving
4. Estimate study days per topic (1 = simple, 3 = complex).
5. recommended_order = dependency-aware order (prerequisites first).
6. subtopics = bullet points or sub-headings found under that topic.
7. Respond strictly with valid JSON. Ensure all property names and string values are enclosed in double quotes. Do not include any text, notes, or markdown fences outside the JSON object.
{{
  "subject": "inferred or provided subject name",
  "topics": [
    {{
      "unit": "Unit 1 / Chapter name",
      "topic": "Topic name",
      "subtopics": ["subtopic 1", "subtopic 2"],
      "estimated_days": 2,
      "difficulty": "medium"
    }}
  ],
  "total_topics": 0,
  "recommended_order": ["Topic A", "Topic B", "Topic C"]
}}
Set total_topics to the count of topics in the array.
"""
    try:
        text = _chat(
            system="You are an expert academic curriculum analyst. Output only valid JSON.",
            history=[],
            user_message=prompt,
            temperature=0.1,
        )
        data = json.loads(_clean_json(text))

        topics = [SyllabusTopic(**t) for t in data.get("topics", [])]
        return SyllabusParseResponse(
            subject=data.get("subject", req.subject or "Unknown"),
            topics=topics,
            total_topics=data.get("total_topics", len(topics)),
            recommended_order=data.get("recommended_order", [t.topic for t in topics]),
        )
    except json.JSONDecodeError as e:
        logger.error("JSON error: %s", e)
        raise HTTPException(status_code=502, detail=f"NIM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NIM error: {e}")
