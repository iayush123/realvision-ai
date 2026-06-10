"""
FastAPI route handlers for RealVision AI.

Endpoints:
  POST /analyze          — Analyze property images (full LangGraph pipeline)
  POST /chat             — Multi-turn conversational Q&A about a property
  POST /recommend        — Rank analyzed properties for a buyer
  GET  /session/{id}     — Session summary & context
  DELETE /session/{id}   — Clear session
  GET  /health           — Health check
"""
import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse

from models.schemas import (
    ImageAnalysisRequest, ChatRequest, RecommendationRequest,
    PropertyAnalysisResult, ChatResponse, RecommendationResponse,
    PropertyScore, BuyerPreferences,
)
from agents.property_agent import analyze_property
from agents.chat_agent import chat_with_property
from tools.property_tools import rank_properties_for_buyer
from memory.session_memory import get_session_store
from config import settings

router = APIRouter()


# ── Analyze ──────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=PropertyAnalysisResult,
    summary="Analyze property images",
    description=(
        "Run the full multimodal AI pipeline: GPT-4o Vision analyzes each image, "
        "features are aggregated, a marketing listing is generated, and valuation "
        "insight is produced. Also stores results in a new session for follow-up chat."
    ),
)
async def analyze_property_endpoint(request: ImageAnalysisRequest) -> PropertyAnalysisResult:
    if not request.image_urls:
        raise HTTPException(status_code=400, detail="At least one image URL is required.")

    if len(request.image_urls) > settings.max_images_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_images_per_request} images per request.",
        )

    try:
        result = await analyze_property(
            image_urls=request.image_urls,
            property_type=request.property_type.value if request.property_type else "house",
            listing_price=request.listing_price,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Auto-store in a session so the user can immediately chat about it
    store = get_session_store()
    store.update_context(result.property_id, {
        "property_analysis": result.model_dump(),
        "images": request.image_urls,
    })

    return result


# ── Chat ─────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat about a property",
    description=(
        "Ask follow-up questions about a previously analyzed property. "
        "Memory persists for the session duration (default 1 hour). "
        "Optionally attach new images to add context."
    ),
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        answer, confidence = await chat_with_property(
            session_id=request.session_id,
            user_message=request.message,
            new_image_urls=request.image_urls,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        referenced_images=request.image_urls or [],
        confidence=confidence,
    )


# ── Recommend ────────────────────────────────────────────────────────────────

@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Rank properties for a buyer",
    description=(
        "Given buyer preferences and a list of previously analyzed property IDs, "
        "rank the properties by AI match score."
    ),
)
async def recommend_endpoint(request: RecommendationRequest) -> RecommendationResponse:
    store = get_session_store()
    prefs = request.buyer_preferences

    # Fetch analyzed properties from session store
    properties: list[dict[str, Any]] = []
    for prop_id in request.candidate_property_ids:
        ctx = store.get_context(prop_id, "property_analysis")
        if ctx:
            properties.append(ctx)

    if not properties:
        raise HTTPException(
            status_code=404,
            detail=(
                "None of the provided property IDs have been analyzed yet. "
                "Run /analyze first."
            ),
        )

    try:
        ranked_json = rank_properties_for_buyer.invoke({
            "properties_json": json.dumps(properties, indent=2),
            "buyer_preferences_json": prefs.model_dump_json(indent=2),
        })
        ranked_data: list[dict] = json.loads(ranked_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {e}")

    ranked = [
        PropertyScore(
            property_id=r["property_id"],
            match_score=r["match_score"],
            match_reasons=r.get("match_reasons", []),
            concerns=r.get("concerns", []),
        )
        for r in ranked_data
    ]
    ranked.sort(key=lambda x: x.match_score, reverse=True)

    top = ranked[0] if ranked else None
    summary_parts = [
        f"Evaluated {len(ranked)} properties. "
        f"Top pick: {top.property_id} (score {top.match_score}/10)." if top else "No matching properties found."
    ]
    if top and top.match_reasons:
        summary_parts.append(f"Why: {top.match_reasons[0]}")

    return RecommendationResponse(
        ranked_properties=ranked,
        recommendation_summary=" ".join(summary_parts),
        top_pick_id=top.property_id if top else "",
    )


# ── Session Management ───────────────────────────────────────────────────────

@router.get(
    "/session/{session_id}",
    summary="Get session info",
)
async def get_session(session_id: str) -> dict[str, Any]:
    store = get_session_store()
    summary = store.session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return summary


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(session_id: str) -> None:
    store = get_session_store()
    store.delete_session(session_id)


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    store = get_session_store()
    return {
        "status": "ok",
        "version": settings.version,
        "active_sessions": str(store.session_count()),
    }
