# Smart FAQ Answering System

TF-IDF + Claude LLM based FAQ retrieval system with hallucination control.

## Features
- `POST /ask` — Finds the best FAQ match for a user query
- TF-IDF + Cosine Similarity based retrieval
- Confidence score with fallback handling
- Optional Claude LLM for natural language answers (grounded only)
- Runtime LLM toggle without server restart
- Basic logging and monitoring

---

## Project Structure

```
faq-system/
├── main.py              ← FastAPI app + all endpoints
├── retrieval.py         ← TF-IDF retrieval logic
├── llm.py               ← Claude LLM integration (optional)
├── knowledge_base.py    ← FAQ data
├── tests/
│   ├── __init__.py
│   └── test_main.py     ← All endpoint tests
├── requirements.txt     ← Python dependencies
├── render.yaml          ← Render.com deployment config
├── .env.example         ← Environment variables template
├── .gitignore
└── README.md
```

---

## Local Setup

```bash
# 1. Go into project folder
cd faq-system

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY

# 6. Start server
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## Running Tests

### Install test dependencies
```bash
pip install pytest httpx
```

### Run all tests
```bash
pytest tests/ -v
```

### Run a specific test class
```bash
pytest tests/test_main.py::TestAskEndpoint -v
pytest tests/test_main.py::TestLLMToggleEndpoint -v
pytest tests/test_main.py::TestFAQsEndpoint -v
```

### Run a specific test case
```bash
pytest tests/test_main.py::TestAskEndpoint::test_ask_exact_match_query -v
```

### Run with print output visible
```bash
pytest tests/ -v -s
```

### Run and stop at first failure
```bash
pytest tests/ -v -x
```

---

## API Endpoints

### POST /ask
Find an answer for a user query.

**Request:**
```json
{
  "query": "I forgot my password, what do I do?"
}
```

**Response (match found):**
```json
{
  "answer": "To reset your password, go to Settings and click Reset Password. You will receive an email with further instructions.",
  "confidence": 0.156,
  "source": ["1"],
  "fallback": false
}
```

**Response (no match):**
```json
{
  "answer": "I don't have enough information to answer this.",
  "confidence": 0.08,
  "source": [],
  "fallback": true
}
```

---

### GET /llm-status
Check whether LLM is currently active or not.

**Response:**
```json
{
  "llm_enabled": true,
  "status": "active",
  "message": "LLM is enabled. Answers are being enhanced by Claude AI."
}
```

---

### POST /llm-toggle
Enable or disable LLM at runtime without restarting the server.

**Request:**
```json
{ "enabled": false }
```

**Response:**
```json
{
  "previous_status": true,
  "current_status": false,
  "message": "LLM has been disabled. Direct FAQ answers will be returned."
}
```

---

### GET /faqs
Returns all FAQs in the knowledge base.

```json
{
  "total": 5,
  "faqs": [...]
}
```

---

### GET /health
Health check endpoint.

```json
{ "status": "ok", "llm_enabled": true }
```

---

## Test with curl

```bash
# Password reset query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "I forgot my password, what do I do?"}'

# Refund query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long does refund take?"}'

# Out-of-scope query (fallback test)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?"}'

# Check LLM status
curl http://localhost:8000/llm-status

# Disable LLM
curl -X POST http://localhost:8000/llm-toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Only if USE_LLM=true | — | Claude API key from console.anthropic.com |
| `USE_LLM` | No | `true` | Enable or disable LLM |

---

## How It Works

```
User Query
    ↓
TF-IDF Vectorizer → Cosine Similarity against FAQ question + answer
    ↓
Best match score > 0.15 threshold?
    ├── YES → Retrieved FAQ answer
    │           ↓
    │       LLM enabled? → Claude rephrases answer (strictly grounded)
    │           ↓
    │       Return: answer + confidence + source + fallback=false
    │
    └── NO  → Return: "I don't have enough info" + fallback=true
```

---

## Deploy on Render (Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect GitHub repo
3. Add environment variable `ANTHROPIC_API_KEY` in Render dashboard
4. Click Deploy — live URL will be generated automatically