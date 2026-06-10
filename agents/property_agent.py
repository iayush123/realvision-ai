"""
RealVision LangGraph Agent — multi-step property analysis pipeline.

Graph topology:
  [START]
     │
     ▼
  analyze_images        ← GPT-4o Vision: per-image room analysis
     │
     ▼
  aggregate_features    ← Merge results, compute overall score
     │
     ▼
  generate_listing      ← GPT-4o Vision: marketing copy from all images
     │
     ▼
  valuation_insight     ← LLM: price assessment
     │
     ▼
  compile_result        ← Build final PropertyAnalysisResult
     │
     ▼
  [END]
"""
import json
import uuid
import asyncio
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from models.schemas import (
    AgentState, RoomAnalysis, RoomType, ConditionRating,
    PropertyAnalysisResult,
)
from tools.vision_tools import analyze_single_image
from tools.property_tools import estimate_property_value
from config import get_openai_client, get_llm


# ── LangGraph State ──────────────────────────────────────────────────────────
# We use TypedDict so LangGraph can serialize/deserialize state cleanly.

class GraphState(TypedDict):
    images: list[str]
    property_type: str
    listing_price: float | None
    property_id: str
    raw_vision_outputs: list[dict[str, Any]]
    extracted_features: dict[str, Any]
    room_analyses: list[dict]          # serialized RoomAnalysis
    generated_listing: str
    valuation_json: str
    final_result: dict | None
    error: str | None


# ── Node implementations ─────────────────────────────────────────────────────

async def analyze_images_node(state: GraphState) -> GraphState:
    """Run GPT-4o Vision on each image concurrently."""
    client = get_openai_client()
    context = f"Property type: {state['property_type']}"
    if state.get("listing_price"):
        context += f", Asking price: ${state['listing_price']:,.0f}"

    tasks = [
        analyze_single_image(client, url, context)
        for url in state["images"]
    ]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        vision_outputs = []
        for r in results:
            if isinstance(r, Exception):
                vision_outputs.append({"error": str(r)})
            else:
                vision_outputs.append(r)
        return {**state, "raw_vision_outputs": vision_outputs}
    except Exception as e:
        return {**state, "error": f"Vision analysis failed: {e}"}


def aggregate_features_node(state: GraphState) -> GraphState:
    """Merge per-image outputs into consolidated features and room list."""
    if state.get("error"):
        return state

    outputs = state["raw_vision_outputs"]
    valid = [o for o in outputs if "error" not in o]

    if not valid:
        return {**state, "error": "All image analyses failed — check image URLs."}

    # Build room analyses
    rooms = []
    all_features: list[str] = []
    styles: list[str] = []
    scores: list[float] = []

    for i, v in enumerate(valid):
        room = RoomAnalysis(
            room_type=RoomType(v.get("room_type", "unknown")),
            condition=ConditionRating(v.get("condition", "fair")),
            quality_score=float(v.get("quality_score", 5.0)),
            detected_features=v.get("detected_features", []),
            improvement_suggestions=v.get("improvement_suggestions", []),
            description=v.get("description", ""),
        )
        rooms.append(room.model_dump())
        all_features.extend(v.get("detected_features", []))
        if v.get("style"):
            styles.append(v["style"])
        scores.append(float(v.get("quality_score", 5.0)))

    overall_score = round(sum(scores) / len(scores), 1) if scores else 5.0
    dominant_style = max(set(styles), key=styles.count) if styles else "unknown"
    unique_features = list(set(all_features))

    extracted = {
        "overall_quality_score": overall_score,
        "dominant_style": dominant_style,
        "all_features": unique_features,
        "total_rooms": len(rooms),
    }

    return {**state, "room_analyses": rooms, "extracted_features": extracted}


async def generate_listing_node(state: GraphState) -> GraphState:
    """Generate marketing listing description from all images."""
    if state.get("error"):
        return state

    client = get_openai_client()
    from tools.vision_tools import vision_generate_listing_description

    try:
        result = await vision_generate_listing_description.ainvoke({
            "image_urls": state["images"],
            "property_type": state["property_type"],
            "listing_price": state.get("listing_price"),
        })
        data = json.loads(result)
        listing_text = (
            f"{data.get('headline', '')}\n\n"
            f"{data.get('description', '')}\n\n"
            "Key Features:\n" +
            "\n".join(f"• {f}" for f in data.get("key_features", [])) +
            f"\n\n{data.get('call_to_action', '')}"
        )
        return {**state, "generated_listing": listing_text}
    except Exception as e:
        # Non-fatal: continue without listing
        return {**state, "generated_listing": f"[Listing generation failed: {e}]"}


def valuation_insight_node(state: GraphState) -> GraphState:
    """Use LLM tool to generate valuation insight."""
    if state.get("error"):
        return state

    features = state["extracted_features"].get("all_features", [])
    try:
        result = estimate_property_value.invoke({
            "quality_score": state["extracted_features"]["overall_quality_score"],
            "detected_features": features,
            "property_type": state["property_type"],
            "listing_price": state.get("listing_price"),
        })
        return {**state, "valuation_json": result}
    except Exception as e:
        return {**state, "valuation_json": json.dumps({"error": str(e)})}


def compile_result_node(state: GraphState) -> GraphState:
    """Assemble final PropertyAnalysisResult."""
    if state.get("error"):
        return state

    ext = state["extracted_features"]
    rooms = [RoomAnalysis(**r) for r in state["room_analyses"]]
    features = ext.get("all_features", [])

    # Parse valuation
    try:
        val_data = json.loads(state.get("valuation_json", "{}"))
        val_insight = (
            f"Verdict: {val_data.get('valuation_verdict', 'N/A')}. "
            f"{val_data.get('price_assessment', '')} "
            f"Tip: {val_data.get('negotiation_tip', '')}"
        )
    except Exception:
        val_insight = "Valuation data unavailable."

    # Key selling points = top 5 features from highest-scoring rooms
    sorted_rooms = sorted(rooms, key=lambda r: r.quality_score, reverse=True)
    key_points = []
    for r in sorted_rooms[:3]:
        key_points.extend(r.detected_features[:2])
    key_points = list(dict.fromkeys(key_points))[:6]  # dedupe, cap at 6

    # Concerns = improvement suggestions from low-scoring rooms
    concerns = []
    for r in sorted_rooms[::-1][:2]:
        concerns.extend(r.improvement_suggestions[:2])
    concerns = list(dict.fromkeys(concerns))[:4]

    result = PropertyAnalysisResult(
        property_id=state["property_id"],
        overall_quality_score=ext["overall_quality_score"],
        estimated_style=ext["dominant_style"],
        total_rooms_detected=ext["total_rooms"],
        rooms=rooms,
        ai_generated_listing=state.get("generated_listing", ""),
        key_selling_points=key_points,
        potential_concerns=concerns,
        valuation_insight=val_insight,
        raw_analysis={"extracted_features": ext},
    )

    return {**state, "final_result": result.model_dump()}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_property_analysis_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("analyze_images", analyze_images_node)
    graph.add_node("aggregate_features", aggregate_features_node)
    graph.add_node("generate_listing", generate_listing_node)
    graph.add_node("valuation_insight", valuation_insight_node)
    graph.add_node("compile_result", compile_result_node)

    graph.set_entry_point("analyze_images")
    graph.add_edge("analyze_images", "aggregate_features")
    graph.add_edge("aggregate_features", "generate_listing")
    graph.add_edge("generate_listing", "valuation_insight")
    graph.add_edge("valuation_insight", "compile_result")
    graph.add_edge("compile_result", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

async def analyze_property(
    image_urls: list[str],
    property_type: str = "house",
    listing_price: float | None = None,
) -> PropertyAnalysisResult:
    """
    Run the full LangGraph property analysis pipeline.

    Args:
        image_urls:     List of image URLs for the property.
        property_type:  'house', 'apartment', 'condo', etc.
        listing_price:  Optional asking price in USD.

    Returns:
        PropertyAnalysisResult with all analysis fields populated.
    """
    graph = build_property_analysis_graph()
    property_id = f"prop_{uuid.uuid4().hex[:8]}"

    initial_state: GraphState = {
        "images": image_urls,
        "property_type": property_type,
        "listing_price": listing_price,
        "property_id": property_id,
        "raw_vision_outputs": [],
        "extracted_features": {},
        "room_analyses": [],
        "generated_listing": "",
        "valuation_json": "",
        "final_result": None,
        "error": None,
    }

    final_state = await graph.ainvoke(initial_state)

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    return PropertyAnalysisResult(**final_state["final_result"])
