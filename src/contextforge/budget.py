from __future__ import annotations

import tiktoken

from contextforge.core.logger import get_logger

log = get_logger(__name__)

# Note: cl100k_base is an approximation for Anthropic.
# Anthropic does not publish their exact tokenizer.
# Error margin: typically ±5% vs actual Anthropic token counts.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class TokenCounter:
    """Provider-aware token estimation using tiktoken cl100k_base.

    IMPORTANT: This is an approximation for Anthropic models.
    Actual Anthropic token counts may differ by ~5%.
    Use for budget enforcement, not for exact billing estimates.
    """

    def __init__(self, provider: str = "anthropic") -> None:
        self._provider = provider

    def count(self, text: str) -> int:
        if not text:
            return 0
        count = len(_ENCODING.encode(text))
        log.debug("token_count", provider=self._provider, count=count, note="approximate")
        return count


class BudgetAllocator:
    """Distributes a total token budget across sources proportional to relevance scores."""

    def __init__(self, min_tokens: int = 50) -> None:
        self._min = min_tokens

    def allocate(self, scores: list[float], total_budget: int) -> list[int]:
        if not scores:
            return []
        total = sum(scores)
        if total == 0:
            equal = total_budget // len(scores)
            return [equal] * len(scores)
        budgets = [max(int((s / total) * total_budget), self._min) for s in scores]
        # Trim proportionally if minimums pushed over budget
        actual = sum(budgets)
        if actual > total_budget:
            excess = actual - total_budget
            for i in range(len(budgets) - 1, -1, -1):
                trim = min(budgets[i] - self._min, excess)
                budgets[i] -= trim
                excess -= trim
                if excess <= 0:
                    break
        return budgets
