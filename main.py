# ============================================================
#  EduPlatform Backend — Enhanced Debug Version
#  Stack : FastAPI + Gemini 1.5 Flash
# ============================================================

import base64
import json
import re
import logging
from datetime import datetime as dt
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Configure logging to see details in Render logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EduPlatform")

# ╔══════════════════════════════════════════════════════════╗
# ║  1. CONFIG                                               ║
# ╚══════════════════════════════════════════════════════════╝

class Settings(BaseSettings):
    gemini_api_key: str = ""
    allowed_origin: str = "*"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# ╔══════════════════════════════════════════════════════════╗
# ║  2. GEMINI HELPERS                                       ║
# ╚══════════════════════════════════════════════════════════╝

# Global variable to hold the dynamically discovered model name
MODEL_NAME = "gemini-1.5-flash" 

def _model(temperature: float = 0.0):
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024,
        )
    )

def _model_large(temperature: float = 0.3):
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            max_output_tokens=4096,
        )
    )

def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _teacher_persona(subject: str, topic: str, mode: str = "text") -> str:
    return f"You are an expert teacher of {subject} on the topic {topic}."

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
    image_mime: Optional[str] = "image/jpeg"
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
# ║  4. APP                                                  ║
# ╚══════════════════════════════════════════════════════════╝

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allowed_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    if not settings.gemini_api_key:
        logger.error("FATAL: GEMINI_API_KEY is not set. Add it to .env or Railway env vars.")
    
    genai.configure(api_key=settings.gemini_api_key)
    
    # FIX: List all available models and PICK THE BEST STABLE ONE
    global MODEL_NAME
    try:
        logger.info("--- DISCOVERING MODELS ---")
        best_match = None
        for m in genai.list_models():
            logger.info(f"Key has access to: {m.name}")
            if "generateContent" in m.supported_generation_methods:
                name = m.name
                # Prioritize Flash models, but avoid "preview", "live", or "lite" for stability
                if "flash" in name.lower():
                    if "preview" not in name.lower() and "live" not in name.lower() and "lite" not in name.lower():
                        best_match = name
                        break # Found a perfect stable Flash model (e.g., gemini-2.5-flash)
                    elif best_match is None:
                        best_match = name # Fallback to any Flash model
                elif best_match is None:
                    best_match = name # Fallback to anything that supports generation
        
        if best_match:
            MODEL_NAME = best_match
        
        logger.info(f"--- AUTO-SELECTED STABLE MODEL: {MODEL_NAME} ---")
    except Exception as e:
        logger.error(f"Failed to discover models: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy"}

# ╔══════════════════════════════════════════════════════════╗
# ║  5. ROUTES                                               ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/syllabus/parse", response_model=SyllabusResponse)
async def parse_syllabus(req: SyllabusRequest):
    logger.info(f"Incoming /syllabus/parse for subject: {req.subject}")
    
    prompt_text = """Parse the provided syllabus. Respond ONLY with valid JSON:
    {
      "topics": ["topic1", "topic2"],
      "estimated_total_hours": 10.0,
      "topic_details": [{"topic": "topic1", "estimated_min": 300, "complexity": "standard"}]
    }"""

    try:
        if req.image_base64:
            logger.info(f"Decoding image base64 (size: {len(req.image_base64)})")
            image_data = base64.b64decode(req.image_base64)
            response = _model_large(temperature=0.0).generate_content([
                {"mime_type": req.image_mime, "data": image_data},
                prompt_text,
            ])
        elif req.pdf_base64:
            logger.info(f"Decoding PDF base64 (size: {len(req.pdf_base64)})")
            pdf_data = base64.b64decode(req.pdf_base64)
            response = _model_large(temperature=0.0).generate_content([
                {"mime_type": "application/pdf", "data": pdf_data},
                prompt_text,
            ])
        elif req.raw_text:
            logger.info(f"Processing raw text (length: {len(req.raw_text)})")
            full_prompt = f"{prompt_text}\n\nSYLLABUS TEXT:\n{req.raw_text}"
            response = _model_large(temperature=0.0).generate_content(full_prompt)
        else:
            logger.warning("No content provided in request")
            raise HTTPException(status_code=400, detail="Provide raw_text, image_base64, or pdf_base64.")

        # Check if response was blocked by safety filters
        if not response.candidates:
            logger.error(f"AI Blocked Response. Feedback: {response.prompt_feedback}")
            raise HTTPException(status_code=502, detail="AI Safety filter blocked the syllabus content.")

        raw_ai_text = response.text
        logger.info(f"Raw AI Response: {raw_ai_text}")

        try:
            data = json.loads(_clean_json(raw_ai_text))
        except Exception as json_err:
            logger.error(f"JSON Parse Error: {str(json_err)} | Raw Text: {raw_ai_text}")
            raise HTTPException(status_code=502, detail="AI returned invalid JSON format.")
        return SyllabusResponse(
            topics=data.get("topics", []),
            estimated_total_hours=data.get("estimated_total_hours", 0.0),
            topic_details=data.get("topic_details", []),
        )

    except Exception as e:
        logger.error(f"FAILED /syllabus/parse: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Gemini processing error: {str(e)}")

@app.post("/plan/generate", response_model=PlanResponse)
async def generate_plan(req: PlanRequest):
    logger.info(f"Generating plan for {req.subject}")
    # Simplified logic to ensure it doesn't crash
    daily_hours = 2.0
    days = 7
    prompt = f"Create a study plan for {req.subject} with topics {req.topics}. Return JSON."
    
    try:
        response = _model_large(temperature=0.3).generate_content(prompt)
        raw_text = response.text
        logger.info(f"Raw AI Plan: {raw_text}")
        data = json.loads(_clean_json(raw_text))
        
        return PlanResponse(
            schedule=data.get("schedule", []),
            summary=data.get("summary", "Plan generated."),
            total_days=data.get("total_days", days),
            daily_hours_recommended=daily_hours,
        )
    except Exception as e:
        logger.error(f"FAILED /plan/generate: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Plan error: {str(e)}")

# ... (Add other endpoints similarly with logger.info for debugging) ...

@app.post("/lecture/intro", response_model=LectureIntroResponse)
async def lecture_intro(req: LectureIntroRequest):
    try:
        prompt = f"Write a 200 word introduction to {req.topic} in {req.subject}."
        reply = _model(temperature=0.0).generate_content(prompt).text.strip()
        return LectureIntroResponse(intro=reply)
    except Exception as e:
        logger.error(f"FAILED /lecture/intro: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/lecture/chat", response_model=LectureChatResponse)
async def lecture_chat(req: LectureChatRequest):
    try:
        # Mock logic for simple test
        user_msg_count = len(req.history)
        progress = min(95.0, (user_msg_count / 20) * 100)
        
        prompt = f"Topic: {req.topic}. Student says: {req.message}. Reply briefly."
        reply = _model(temperature=0.0).generate_content(prompt).text.strip()
        
        return LectureChatResponse(reply=reply, phase="TEACHING", progress_pct=progress)
    except Exception as e:
        logger.error(f"FAILED /lecture/chat: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/lecture/summary", response_model=SummaryResponse)
async def lecture_summary(req: SummaryRequest):
    try:
        prompt = f"Summarize the lecture on {req.topic}."
        summary = _model(temperature=0.0).generate_content(prompt).text.strip()
        return SummaryResponse(summary=summary)
    except Exception as e:
        logger.error(f"FAILED /lecture/summary: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/doubt/ask", response_model=DoubtResponse)
async def ask_doubt(req: DoubtRequest):
    try:
        prompt = f"Answer doubt about {req.topic}: {req.question}"
        answer = _model(temperature=0.0).generate_content(prompt).text.strip()
        return DoubtResponse(answer=answer)
    except Exception as e:
        logger.error(f"FAILED /doubt/ask: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))
