"""
Pydantic schemas for RealVision AI
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PropertyType(str, Enum):
    APARTMENT = "apartment"
    HOUSE = "house"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    COMMERCIAL = "commercial"


class RoomType(str, Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    DINING_ROOM = "dining_room"
    OUTDOOR = "outdoor"
    GARAGE = "garage"
    UNKNOWN = "unknown"


class ConditionRating(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    NEEDS_WORK = "needs_work"


# ── Request Models ──────────────────────────────────────────────────────────

class ImageAnalysisRequest(BaseModel):
    image_urls: List[str] = Field(..., description="List of property image URLs to analyze")
    property_type: Optional[PropertyType] = Field(None, description="Type of property")
    listing_price: Optional[float] = Field(None, description="Listed price in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "image_urls": ["https://example.com/property1.jpg"],
                "property_type": "apartment",
                "listing_price": 450000
            }
        }


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for memory continuity")
    message: str = Field(..., description="User's question about the property")
    image_urls: Optional[List[str]] = Field(None, description="Optional new images to reference")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "message": "What's the condition of the kitchen?",
                "image_urls": None
            }
        }


class BuyerPreferences(BaseModel):
    budget_min: float = Field(..., description="Minimum budget in USD")
    budget_max: float = Field(..., description="Maximum budget in USD")
    preferred_rooms: List[RoomType] = Field(default_factory=list)
    min_quality_score: float = Field(default=6.0, ge=0, le=10)
    style_preferences: List[str] = Field(default_factory=list, description="e.g. modern, rustic, minimalist")
    must_have_features: List[str] = Field(default_factory=list, description="e.g. pool, garage, open kitchen")


class RecommendationRequest(BaseModel):
    buyer_preferences: BuyerPreferences
    candidate_property_ids: List[str] = Field(..., description="IDs of analyzed properties to score")


# ── Response Models ─────────────────────────────────────────────────────────

class RoomAnalysis(BaseModel):
    room_type: RoomType
    condition: ConditionRating
    quality_score: float = Field(..., ge=0, le=10, description="AI-assigned quality score 0-10")
    detected_features: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    description: str


class PropertyAnalysisResult(BaseModel):
    property_id: str
    overall_quality_score: float = Field(..., ge=0, le=10)
    estimated_style: str
    total_rooms_detected: int
    rooms: List[RoomAnalysis]
    ai_generated_listing: str = Field(..., description="Auto-generated marketing description")
    key_selling_points: List[str]
    potential_concerns: List[str]
    valuation_insight: str
    raw_analysis: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    referenced_images: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)


class PropertyScore(BaseModel):
    property_id: str
    match_score: float = Field(..., ge=0, le=10)
    match_reasons: List[str]
    concerns: List[str]


class RecommendationResponse(BaseModel):
    ranked_properties: List[PropertyScore]
    recommendation_summary: str
    top_pick_id: str


# ── Internal State (LangGraph) ───────────────────────────────────────────────

class AgentState(BaseModel):
    """Shared state passed through LangGraph nodes"""
    images: List[str] = Field(default_factory=list)
    raw_vision_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_features: Dict[str, Any] = Field(default_factory=dict)
    room_analyses: List[RoomAnalysis] = Field(default_factory=list)
    generated_listing: str = ""
    valuation_context: str = ""
    final_result: Optional[PropertyAnalysisResult] = None
    error: Optional[str] = None
    property_id: str = ""
    listing_price: Optional[float] = None
