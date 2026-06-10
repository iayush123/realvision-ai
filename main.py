"""
RealVision AI — FastAPI entry point.

Run locally:
    uvicorn main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from memory.session_memory import get_session_store
from config import settings


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: kick off a background task to purge expired sessions every 10 min
    store = get_session_store()
    purge_task = asyncio.create_task(_purge_loop(store))
    yield
    purge_task.cancel()


async def _purge_loop(store):
    """Background coroutine: purge expired sessions every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        n = store.purge_expired()
        if n:
            print(f"[SessionStore] Purged {n} expired session(s).")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RealVision AI",
    description=(
        "## Multimodal Real Estate Intelligence Platform\n\n"
        "Analyzes property images using GPT-4o Vision to generate:\n"
        "- Room-by-room quality scores and condition reports\n"
        "- AI-generated marketing listings\n"
        "- Valuation insights\n"
        "- Buyer preference matching & recommendations\n"
        "- Conversational Q&A with session memory\n\n"
        "**Tech stack:** FastAPI · LangGraph · LangChain · GPT-4o Vision · OpenAI API"
    ),
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1", tags=["RealVision AI"])


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


from fastapi.responses import FileResponse

@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse("frontend.html")