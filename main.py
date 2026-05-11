# main.py
# FastAPI app — /ask endpoint yahan hai

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from retrieval import find_best_match

load_dotenv()

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                        # Console output
        logging.FileHandler("faq_system.log")          # File mein bhi save hoga
    ]
)
logger = logging.getLogger(__name__)

# ─── Use LLM ya nahi? ─────────────────────────────────────────────────────────
# True  → Claude se better answer milega (ANTHROPIC_API_KEY chahiye)
# False → Direct FAQ answer return hoga (no API needed)
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart FAQ Answering System",
    description="TF-IDF + LLM based FAQ retrieval with hallucination control",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, example="I forgot my password, what do I do?")


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    source: list[str]
    fallback: bool

class LLMToggleRequest(BaseModel):
    model_config = ConfigDict(strict=True)  # ← yeh add karo
    enabled: bool = Field(..., example=True)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Smart FAQ API is running!",
        "docs": "/docs",
        "ask_endpoint": "POST /ask"
    }


@app.get("/health")
async def health():
    """Health check — deployment monitoring ke liye"""
    return {"status": "ok", "llm_enabled": USE_LLM}


@app.get("/llm-status")
async def llm_status():
    """
    LLM ka current status check karo.
    Batayega ki LLM enable hai ya disable, aur kyun.
    """
    return {
        "llm_enabled": USE_LLM,
        "status": "active" if USE_LLM else "inactive",
        "message": (
            "LLM is enabled. Answers are being enhanced by Claude AI."
            if USE_LLM else
            "LLM is disabled. Returning direct FAQ answers only."
        )
    }


@app.post("/llm-toggle")
async def llm_toggle(request: LLMToggleRequest):
    """
    LLM ko runtime mein enable ya disable karo.
    Server restart ki zaroorat nahi!
 
    Request body:
    - enabled: true  -> LLM on karo
    - enabled: false -> LLM off karo
    """
    global USE_LLM
    previous = USE_LLM
    USE_LLM = request.enabled
 
    logger.info(f"LLM status changed: {previous} -> {USE_LLM}")
 
    return {
        "previous_status": previous,
        "current_status": USE_LLM,
        "message": (
            "LLM has been enabled. Claude AI will now enhance answers."
            if USE_LLM else
            "LLM has been disabled. Direct FAQ answers will be returned."
        )
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    User query ke basis par FAQ se answer dhundta hai.
    
    - TF-IDF + Cosine Similarity se best FAQ match karta hai
    - Confidence score return karta hai
    - Low confidence → fallback response
    - LLM se answer ko natural language mein rephrase karta hai (if enabled)
    """

    logger.info(f"Query received: '{request.query}'")

    # Step 1: TF-IDF retrieval
    result = find_best_match(request.query)

    # Step 2: Agar fallback nahi hai aur LLM enabled hai → better answer generate karo
    if USE_LLM and not result["fallback"] and result.get("_retrieved_faq"):
        try:
            from llm import generate_grounded_answer
            llm_answer = generate_grounded_answer(request.query, result["_retrieved_faq"])
            result["answer"] = llm_answer
            logger.info(f"LLM answer generated successfully.")
        except Exception as e:
            logger.warning(f"LLM failed, using direct FAQ answer. Error: {e}")

    # Step 3: Internal key remove karo before returning
    result.pop("_retrieved_faq", None)

    logger.info(
        f"Response - confidence={result['confidence']}, "
        f"fallback={result['fallback']}, source={result['source']}"
    )

    return result


@app.get("/faqs")
async def list_faqs():
    """Saare available FAQs dekho"""
    from knowledge_base import FAQ_DATA
    return {"total": len(FAQ_DATA), "faqs": FAQ_DATA}
