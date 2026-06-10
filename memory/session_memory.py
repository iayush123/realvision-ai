"""
Session Memory — in-memory store for multi-turn property conversations.

Stores per-session: chat history, analyzed images, and property context.
In production, swap _store for Redis or a vector DB.
"""
import time
from collections import defaultdict
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class SessionStore:
    """
    Thread-safe (for single-process use) in-memory session store.

    Each session holds:
        messages   : LangChain message history
        context    : Arbitrary dict (property analysis results, images, etc.)
        created_at : Unix timestamp
        last_active: Unix timestamp
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, dict[str, Any]] = {}
        self.ttl = ttl_seconds

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def create_session(self, session_id: str) -> None:
        now = time.time()
        self._store[session_id] = {
            "messages": [],
            "context": {},
            "created_at": now,
            "last_active": now,
        }

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._store:
            self.create_session(session_id)
        self._touch(session_id)
        return self._store[session_id]

    def delete_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def _touch(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id]["last_active"] = time.time()

    def purge_expired(self) -> int:
        now = time.time()
        expired = [
            sid for sid, data in self._store.items()
            if now - data["last_active"] > self.ttl
        ]
        for sid in expired:
            del self._store[sid]
        return len(expired)

    # ── Message history ──────────────────────────────────────────────────────

    def add_user_message(self, session_id: str, content: str) -> None:
        sess = self.get_or_create(session_id)
        sess["messages"].append(HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str) -> None:
        sess = self.get_or_create(session_id)
        sess["messages"].append(AIMessage(content=content))

    def add_system_message(self, session_id: str, content: str) -> None:
        sess = self.get_or_create(session_id)
        sess["messages"].append(SystemMessage(content=content))

    def get_messages(self, session_id: str, last_n: int = 20) -> list[BaseMessage]:
        sess = self.get_or_create(session_id)
        return sess["messages"][-last_n:]

    def clear_messages(self, session_id: str) -> None:
        sess = self.get_or_create(session_id)
        sess["messages"] = []

    # ── Context (property data, images, etc.) ────────────────────────────────

    def set_context(self, session_id: str, key: str, value: Any) -> None:
        sess = self.get_or_create(session_id)
        sess["context"][key] = value

    def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        sess = self.get_or_create(session_id)
        return sess["context"].get(key, default)

    def update_context(self, session_id: str, data: dict[str, Any]) -> None:
        sess = self.get_or_create(session_id)
        sess["context"].update(data)

    def get_full_context(self, session_id: str) -> dict[str, Any]:
        sess = self.get_or_create(session_id)
        return sess["context"]

    # ── Introspection ────────────────────────────────────────────────────────

    def session_count(self) -> int:
        return len(self._store)

    def session_summary(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._store:
            return {}
        sess = self._store[session_id]
        return {
            "session_id": session_id,
            "message_count": len(sess["messages"]),
            "context_keys": list(sess["context"].keys()),
            "created_at": sess["created_at"],
            "last_active": sess["last_active"],
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_store = SessionStore(ttl_seconds=3600)


def get_session_store() -> SessionStore:
    return _store
