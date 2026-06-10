"""
Conversational Agent — multi-turn chat about a property with persistent memory.

Uses LangChain's ChatOpenAI with a system prompt that's dynamically enriched
with the property context stored in the session.
"""
import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from memory.session_memory import get_session_store
from tools.vision_tools import _load_image_as_b64, _build_image_message


SYSTEM_TEMPLATE = """You are RealVision AI, an expert real estate assistant.
You have analyzed the following property and have deep knowledge of it.

Property Summary:
{property_summary}

Guidelines:
- Answer questions based on the property analysis data above.
- If you haven't analyzed certain aspects, say so honestly.
- Be specific: reference actual detected features, scores, and conditions.
- Suggest relevant questions the buyer might want to ask.
- Keep responses concise but informative (2–4 sentences unless detail is needed).
"""


def _build_property_summary(context: dict[str, Any]) -> str:
    """Convert session context to a readable property summary for the system prompt."""
    analysis = context.get("property_analysis")
    if not analysis:
        images = context.get("images", [])
        return f"No formal analysis run yet. {len(images)} image(s) uploaded."

    lines = [
        f"Overall Quality Score: {analysis.get('overall_quality_score', 'N/A')}/10",
        f"Estimated Style: {analysis.get('estimated_style', 'N/A')}",
        f"Rooms Detected: {analysis.get('total_rooms_detected', 'N/A')}",
        f"Key Selling Points: {', '.join(analysis.get('key_selling_points', []))}",
        f"Potential Concerns: {', '.join(analysis.get('potential_concerns', []))}",
        f"Valuation Insight: {analysis.get('valuation_insight', 'N/A')}",
    ]

    rooms = analysis.get("rooms", [])
    if rooms:
        lines.append("\nRoom Details:")
        for r in rooms[:5]:
            lines.append(
                f"  - {r['room_type'].replace('_', ' ').title()}: "
                f"{r['condition']} condition, score {r['quality_score']}/10. "
                f"Features: {', '.join(r.get('detected_features', [])[:4])}"
            )

    return "\n".join(lines)


async def chat_with_property(
    session_id: str,
    user_message: str,
    new_image_urls: list[str] | None = None,
) -> tuple[str, float]:
    """
    Continue a multi-turn conversation about a property.

    Args:
        session_id:      Session ID (memory key).
        user_message:    The user's question or message.
        new_image_urls:  Optional new images to add to context.

    Returns:
        (answer_text, confidence_score)
    """
    store = get_session_store()
    session = store.get_or_create(session_id)

    # Optionally register new images in context
    if new_image_urls:
        existing = store.get_context(session_id, "images", [])
        store.set_context(session_id, "images", existing + new_image_urls)

    # Build dynamic system prompt
    context = store.get_full_context(session_id)
    property_summary = _build_property_summary(context)
    system_content = SYSTEM_TEMPLATE.format(property_summary=property_summary)

    # Retrieve recent message history
    history = store.get_messages(session_id, last_n=16)

    # Build the message list for this turn
    messages: list[Any] = [SystemMessage(content=system_content)]
    messages.extend(history)

    # If the user attached new images, build a multimodal message
    if new_image_urls:
        parts: list[dict] = [{"type": "text", "text": user_message}]
        for url in new_image_urls[:4]:
            try:
                mt, b64 = await _load_image_as_b64(url)
                parts.append(_build_image_message(mt, b64))
            except Exception:
                pass  # Skip unloadable images
        messages.append(HumanMessage(content=parts))
    else:
        messages.append(HumanMessage(content=user_message))

    # Call GPT-4o
    llm = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=600)
    response = await llm.ainvoke(messages)
    answer = response.content

    # Persist turn to memory
    store.add_user_message(session_id, user_message)
    store.add_ai_message(session_id, answer)

    # Rough confidence heuristic: full context = higher confidence
    confidence = 0.9 if context.get("property_analysis") else 0.6

    return answer, confidence
