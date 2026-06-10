"""
Vision Tools — GPT-4o Vision wrappers for property image analysis.

Each function is also registered as a LangChain Tool so the agent
can call them autonomously.
"""
import json
import base64
from typing import Any
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from langchain_core.tools import tool

from models.schemas import RoomAnalysis, RoomType, ConditionRating


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _load_image_as_b64(source: str) -> tuple[str, str]:
    """
    Accept a URL or a local file path and return (media_type, base64_data).
    """
    if source.startswith("http://") or source.startswith("https://"):
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(source)
            resp.raise_for_status()
            content = resp.content
            media_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]
    else:
        path = Path(source)
        content = path.read_bytes()
        suffix_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".png": "image/png", ".webp": "image/webp"}
        media_type = suffix_map.get(path.suffix.lower(), "image/jpeg")

    return media_type, base64.b64encode(content).decode()


def _build_image_message(media_type: str, b64: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "high"},
    }


# ── Core vision call ─────────────────────────────────────────────────────────

async def analyze_single_image(
    client: AsyncOpenAI,
    image_source: str,
    extra_context: str = "",
) -> dict[str, Any]:
    """
    Send one image to GPT-4o Vision and get structured JSON back.
    Returns raw dict with keys: room_type, condition, quality_score,
    detected_features, improvement_suggestions, description.
    """
    media_type, b64 = await _load_image_as_b64(image_source)

    system_prompt = """You are an expert real estate property analyst with 15 years of
    experience evaluating properties for sale. Analyze the provided property image and
    return ONLY valid JSON — no markdown, no extra text.

    JSON schema:
    {
        "room_type": "<living_room|bedroom|kitchen|bathroom|dining_room|outdoor|garage|unknown>",
        "condition": "<excellent|good|fair|needs_work>",
        "quality_score": <float 0-10>,
        "detected_features": [<string>, ...],
        "improvement_suggestions": [<string>, ...],
        "description": "<2-3 sentence professional description>",
        "style": "<modern|contemporary|traditional|rustic|minimalist|eclectic|unknown>",
        "estimated_sqft_visible": <int or null>,
        "natural_light": "<abundant|moderate|limited>"
    }"""

    user_content: list[dict] = [
        _build_image_message(media_type, b64),
        {"type": "text", "text": f"Analyze this property image.{' Context: ' + extra_context if extra_context else ''}"},
    ]

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=800,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    # Strip possible markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── LangChain Tools ──────────────────────────────────────────────────────────

@tool
async def vision_analyze_room(image_url: str, context: str = "") -> str:
    """
    Analyze a single property room image using GPT-4o Vision.
    Returns a JSON string with room_type, condition, quality_score,
    detected_features, improvement_suggestions, and description.

    Args:
        image_url: URL or local path to the property image.
        context: Optional extra context (e.g. "listing price: $450k").
    """
    from config import get_openai_client  # lazy import to avoid circular deps
    client = get_openai_client()
    result = await analyze_single_image(client, image_url, context)
    return json.dumps(result, indent=2)


@tool
async def vision_compare_rooms(image_urls: list[str], focus: str = "overall quality") -> str:
    """
    Analyze multiple property images together and produce a comparative summary.
    Useful for assessing overall property quality across all rooms.

    Args:
        image_urls: List of image URLs/paths (max 6 recommended).
        focus: What aspect to compare, e.g. 'renovation potential' or 'luxury features'.
    """
    from config import get_openai_client
    client = get_openai_client()

    # Load all images
    image_parts: list[dict] = []
    for src in image_urls[:6]:
        mt, b64 = await _load_image_as_b64(src)
        image_parts.append(_build_image_message(mt, b64))

    image_parts.append({
        "type": "text",
        "text": (
            f"These are multiple images of the same property. "
            f"Focus on: {focus}. "
            "Return JSON with keys: overall_score (0-10), style, key_strengths (list), "
            "key_weaknesses (list), comparable_properties (brief text), "
            "marketing_headline (one punchy sentence)."
        )
    })

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a senior real estate analyst. Return ONLY valid JSON, no markdown."
            },
            {"role": "user", "content": image_parts},
        ],
        max_tokens=1000,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw


@tool
async def vision_generate_listing_description(
    image_urls: list[str],
    property_type: str,
    listing_price: float | None = None,
) -> str:
    """
    Generate a professional, compelling real estate listing description
    from property images using GPT-4o Vision.

    Args:
        image_urls: List of property image URLs.
        property_type: e.g. 'apartment', 'house', 'condo'.
        listing_price: Optional asking price in USD.
    """
    from config import get_openai_client
    client = get_openai_client()

    image_parts: list[dict] = []
    for src in image_urls[:8]:
        mt, b64 = await _load_image_as_b64(src)
        image_parts.append(_build_image_message(mt, b64))

    price_str = f"${listing_price:,.0f}" if listing_price else "price not specified"
    image_parts.append({
        "type": "text",
        "text": (
            f"Write a compelling real estate listing for this {property_type} priced at {price_str}. "
            "Return JSON with: headline (string), description (3-4 paragraphs), "
            "key_features (list of bullet points), call_to_action (string)."
        )
    })

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a top-tier real estate copywriter. "
                    "Write vivid, accurate, persuasive descriptions. "
                    "Return ONLY valid JSON."
                )
            },
            {"role": "user", "content": image_parts},
        ],
        max_tokens=1200,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw
