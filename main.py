# main.py
# FastAPI application — the /ask endpoint is defined here

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
        logging.FileHandler("faq_system.log")          # Will also be saved in the file
    ]
)
logger = logging.getLogger(__name__)

# ─── Use LLM ya nahi? ─────────────────────────────────────────────────────────
# True  → Generates enhanced answers using Claude (requires ANTHROPIC_API_KEY)
# False → Returns direct FAQ answers without using the API
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
    model_config = ConfigDict(strict=True)
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
    """Health check — used for deployment monitoring"""
    return {"status": "ok", "llm_enabled": USE_LLM}


@app.get("/llm-status")
async def llm_status():
    """
    Check the current LLM status.
    Indicates whether the LLM is enabled or disabled, and explains why.
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
    Enable or disable the LLM at runtime.
    No server restart required!

    Request body:
    - enabled: true  -> Enable the LLM
    - enabled: false -> Disable the LLM
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
    Finds the most relevant answer from the FAQ based on the user's query.

    - Uses TF-IDF and Cosine Similarity to identify the best FAQ match
    - Returns a confidence score for the matched result
    - Provides a fallback response for low-confidence matches
    - Rephrases the answer into natural language using the LLM (if enabled)
    """

    logger.info(f"Query received: '{request.query}'")

    # Step 1: TF-IDF retrieval
    result = find_best_match(request.query)

    # Step 2: If the response is not a fallback and the LLM is enabled,
    # generate a more natural and enhanced answer
    if USE_LLM and not result["fallback"] and result.get("_retrieved_faq"):
        try:
            from llm import generate_grounded_answer
            llm_answer = generate_grounded_answer(request.query, result["_retrieved_faq"])
            result["answer"] = llm_answer
            logger.info(f"LLM answer generated successfully.")
        except Exception as e:
            logger.warning(f"LLM failed, using direct FAQ answer. Error: {e}")

    # Step 3: Remove internal keys before returning the response
    result.pop("_retrieved_faq", None)

    logger.info(
        f"Response - confidence={result['confidence']}, "
        f"fallback={result['fallback']}, source={result['source']}"
    )

    return result


@app.get("/faqs")
async def list_faqs():
    """View all available FAQs"""
    from knowledge_base import FAQ_DATA
    return {"total": len(FAQ_DATA), "faqs": FAQ_DATA}
