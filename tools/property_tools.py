"""
Property Tools — non-vision helpers for valuation, feature scoring,
and buyer–property matching.
"""
import json
from typing import Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ── Valuation Tool ───────────────────────────────────────────────────────────

@tool
def estimate_property_value(
    quality_score: float,
    detected_features: list[str],
    property_type: str,
    listing_price: float | None,
    location_hint: str = "USA",
) -> str:
    """
    Generate a valuation insight for a property based on its AI-assessed quality
    score, detected features, and type.

    Args:
        quality_score: Overall quality 0–10 from vision analysis.
        detected_features: Features detected across all images.
        property_type: 'apartment', 'house', 'condo', etc.
        listing_price: Optional current listing price in USD.
        location_hint: General location for market context.

    Returns:
        JSON with keys: valuation_verdict, price_assessment, negotiation_tip,
        investment_outlook.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    features_str = ", ".join(detected_features[:20]) if detected_features else "none specified"
    price_str = f"${listing_price:,.0f}" if listing_price else "undisclosed"

    prompt = f"""You are a real estate valuation expert.

Property details:
- Type: {property_type}
- Location: {location_hint}
- AI Quality Score: {quality_score}/10
- Detected Features: {features_str}
- Listed Price: {price_str}

Return ONLY valid JSON:
{{
    "valuation_verdict": "<fairly_priced|overpriced|underpriced|insufficient_data>",
    "price_assessment": "<2 sentences on price relative to quality and features>",
    "negotiation_tip": "<actionable tip for buyer or seller>",
    "investment_outlook": "<short|medium|long>_term potential note",
    "comparable_range_usd": {{"low": <int>, "high": <int>}} or null
}}"""

    result = llm.invoke(prompt)
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw


# ── Feature Scoring Tool ─────────────────────────────────────────────────────

@tool
def score_features_against_preferences(
    detected_features: list[str],
    must_have_features: list[str],
    nice_to_have_features: list[str],
) -> str:
    """
    Score a property's detected features against buyer preferences.

    Args:
        detected_features: Features found in the property images.
        must_have_features: Features the buyer requires (deal-breakers).
        nice_to_have_features: Features the buyer wants but doesn't require.

    Returns:
        JSON with: must_have_score (0-1), nice_to_have_score (0-1),
        matched_must_haves, matched_nice_to_haves, missing_must_haves.
    """
    detected_lower = {f.lower() for f in detected_features}

    matched_must = []
    missing_must = []
    for feature in must_have_features:
        # Fuzzy substring match
        if any(feature.lower() in d or d in feature.lower() for d in detected_lower):
            matched_must.append(feature)
        else:
            missing_must.append(feature)

    matched_nice = [
        f for f in nice_to_have_features
        if any(f.lower() in d or d in f.lower() for d in detected_lower)
    ]

    must_score = len(matched_must) / len(must_have_features) if must_have_features else 1.0
    nice_score = len(matched_nice) / len(nice_to_have_features) if nice_to_have_features else 1.0

    return json.dumps({
        "must_have_score": round(must_score, 2),
        "nice_to_have_score": round(nice_score, 2),
        "matched_must_haves": matched_must,
        "matched_nice_to_haves": matched_nice,
        "missing_must_haves": missing_must,
    }, indent=2)


# ── Recommendation Ranker Tool ───────────────────────────────────────────────

@tool
def rank_properties_for_buyer(
    properties_json: str,
    buyer_preferences_json: str,
) -> str:
    """
    Given a list of analyzed properties and buyer preferences,
    rank properties by match score using an LLM reasoning pass.

    Args:
        properties_json: JSON array of PropertyAnalysisResult dicts.
        buyer_preferences_json: JSON of BuyerPreferences dict.

    Returns:
        JSON array of {property_id, match_score, match_reasons, concerns},
        sorted by match_score descending.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = f"""You are a real estate recommendation engine.

Buyer preferences:
{buyer_preferences_json}

Properties to evaluate:
{properties_json}

For each property, assign a match_score (0–10) and explain why it does or doesn't
fit the buyer. Be specific.

Return ONLY a valid JSON array:
[
  {{
    "property_id": "<id>",
    "match_score": <float>,
    "match_reasons": ["<reason>", ...],
    "concerns": ["<concern>", ...]
  }},
  ...
]
Sort by match_score descending."""

    result = llm.invoke(prompt)
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw


# ── All tools as a list for agent binding ────────────────────────────────────

ALL_PROPERTY_TOOLS = [
    estimate_property_value,
    score_features_against_preferences,
    rank_properties_for_buyer,
]
