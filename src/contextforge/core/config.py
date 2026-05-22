from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Token counting: uses tiktoken cl100k_base approximation — not exact for Anthropic (~±5%)
    provider: str = "anthropic"
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    benchmark_model: str = "deepseek/deepseek-v4-flash"
    benchmark_input_cost_per_1m: float = 0.14
    benchmark_output_cost_per_1m: float = 0.28
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
