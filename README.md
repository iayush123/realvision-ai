# RealVision AI 🏠

**Multimodal Real Estate Intelligence Platform**

> Built as a portfolio project for the **AI Product Engineer** role at Agent Mira.
> Demonstrates: Agentic AI (LangGraph), Multimodal Intelligence (GPT-4o Vision),
> FastAPI backend, LangChain tooling, Conversational AI with memory, and
> Recommendation systems.

---

## What It Does

RealVision AI takes property listing images and produces:

| Feature | Description |
|---|---|
| **Room Analysis** | GPT-4o Vision scores each room: type, condition, quality (0–10), detected features |
| **Overall Quality Score** | Aggregated across all images |
| **AI-Generated Listing** | Marketing copy (headline, description, bullet points) written by GPT-4o |
| **Valuation Insight** | Fair / overpriced / underpriced verdict with negotiation tips |
| **Chat Q&A** | Ask anything about the property — session memory persists |
| **Buyer Matching** | Score and rank multiple properties against buyer preferences |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│  POST /api/v1/analyze  │  POST /api/v1/chat  │  POST /api/v1/recommend │
└────────────┬───────────┴──────────┬──────────┴─────────┬─────────┘
             │                      │                     │
             ▼                      ▼                     ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │  LangGraph Agent │   │   Chat Agent     │   │  Ranker Tool     │
   │  (property_agent)│   │  (LangChain +    │   │  (LLM-powered)   │
   │                  │   │   session memory)│   │                  │
   │  Nodes:          │   │                  │   └──────────────────┘
   │  1. analyze_imgs ◄───┤  SessionStore    │
   │  2. aggregate    │   │  (in-memory;     │
   │  3. gen_listing  │   │   swap for Redis)│
   │  4. valuation    │   └──────────────────┘
   │  5. compile      │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────────────┐
   │          Tools Layer             │
   │                                  │
   │  vision_tools.py                 │
   │  ┌──────────────────────────┐    │
   │  │  GPT-4o Vision (async)   │    │
   │  │  • analyze_single_image  │    │
   │  │  • vision_compare_rooms  │    │
   │  │  • gen_listing_desc      │    │
   │  └──────────────────────────┘    │
   │                                  │
   │  property_tools.py               │
   │  ┌──────────────────────────┐    │
   │  │  LLM Tools (LangChain)   │    │
   │  │  • estimate_value        │    │
   │  │  • score_features        │    │
   │  │  • rank_properties       │    │
   │  └──────────────────────────┘    │
   └──────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | **FastAPI** + Uvicorn |
| Agentic Pipeline | **LangGraph** (StateGraph) |
| LLM Orchestration | **LangChain** + LangChain-OpenAI |
| Vision Model | **GPT-4o** (multimodal) |
| Text/Tool Model | **GPT-4o-mini** |
| Memory System | In-memory `SessionStore` (drop-in Redis replacement) |
| Schema Validation | **Pydantic v2** |
| Language | **Python 3.11+** |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourname/realvision-ai
cd realvision-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Usage Examples

### Analyze a property

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": [
      "https://example.com/living-room.jpg",
      "https://example.com/kitchen.jpg",
      "https://example.com/bedroom.jpg"
    ],
    "property_type": "apartment",
    "listing_price": 450000
  }'
```

**Response includes:**
- `property_id` — use this as `session_id` for chat
- `overall_quality_score` (e.g. `8.2`)
- `rooms[]` — per-room breakdown
- `ai_generated_listing` — ready-to-publish copy
- `valuation_insight` — fair/overpriced/underpriced

---

### Chat about the property

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "prop_a1b2c3d4",
    "message": "Is the kitchen modern enough for a buyer who loves cooking?"
  }'
```

---

### Get buyer recommendations

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "buyer_preferences": {
      "budget_min": 300000,
      "budget_max": 500000,
      "min_quality_score": 7.0,
      "must_have_features": ["open kitchen", "natural light"],
      "style_preferences": ["modern", "contemporary"]
    },
    "candidate_property_ids": ["prop_a1b2c3d4", "prop_e5f6g7h8"]
  }'
```

---

## Project Structure

```
realvision-ai/
├── main.py                  # FastAPI app entry point
├── config.py                # Settings, OpenAI client factory
├── requirements.txt
├── .env.example
│
├── models/
│   └── schemas.py           # All Pydantic request/response models
│
├── agents/
│   ├── property_agent.py    # LangGraph multi-step analysis pipeline
│   └── chat_agent.py        # Conversational agent with session memory
│
├── tools/
│   ├── vision_tools.py      # GPT-4o Vision LangChain tools
│   └── property_tools.py    # Valuation, feature scoring, ranking tools
│
├── memory/
│   └── session_memory.py    # In-memory session store (swap for Redis)
│
└── api/
    └── routes.py            # FastAPI route handlers
```

---

## Extending the Project

| Idea | How |
|---|---|
| Persistent memory | Replace `SessionStore` with Redis + `aioredis` |
| More agent steps | Add nodes to the LangGraph in `property_agent.py` |
| MCP integration | Expose tools via Model Context Protocol server |
| Vector search | Embed property analyses in Pinecone/Weaviate for semantic search |
| Frontend | React + Tailwind dashboard consuming the REST API |
| Auth | Add FastAPI `Depends` with JWT/API key middleware |

---

## Skills Demonstrated (Agent Mira JD Alignment)

| JD Requirement | Implementation |
|---|---|
| Agentic AI architectures | LangGraph `StateGraph` with 5 sequential nodes |
| Multimodal intelligence | GPT-4o Vision on every image, async concurrent |
| FastAPI backend & scalable APIs | Full REST API with Pydantic validation |
| LangChain / LangGraph | Both used throughout |
| OpenAI APIs | GPT-4o + GPT-4o-mini |
| Recommendation systems | LLM-powered buyer–property matching |
| Conversational AI | Multi-turn chat with persistent session memory |
| MCP / memory systems | `SessionStore` with context propagation |
