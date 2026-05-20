from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Token counting: uses tiktoken cl100k_base approximation — not exact for Anthropic (~±5%)
    provider: str = "anthropic"
    anthropic_api_key: str = ""
    token_budget: int = 8000
    top_k: int = 20
    top_n: int = 5
    scorer_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(
        env_prefix="CONTEXTFORGE_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
