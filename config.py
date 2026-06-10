"""
Shared configuration and dependency factories.
Load environment variables before importing other modules.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

load_dotenv()


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )
    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


class Settings:
    app_name: str = "RealVision AI"
    version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    max_images_per_request: int = int(os.getenv("MAX_IMAGES", "8"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL", "3600"))


settings = Settings()
