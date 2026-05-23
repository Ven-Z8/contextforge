from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    question: str
    answers: list[str]
    contexts: list[str]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class StrategyResult:
    strategy: str
    evidence_hits: int = 0
    recall_at_5_hits: int = 0
    recall_at_10_hits: int = 0
    examples: int = 0
    total_tokens: int = 0
    total_sources: int = 0
    total_mrr: float = 0.0
    total_ndcg_10: float = 0.0
    utilizations: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    @property
    def evidence_recall(self) -> float:
        return self.evidence_hits / self.examples if self.examples else 0.0

    @property
    def recall_at_5(self) -> float:
        return self.recall_at_5_hits / self.examples if self.examples else 0.0

    @property
    def recall_at_10(self) -> float:
        return self.recall_at_10_hits / self.examples if self.examples else 0.0

    @property
    def mrr(self) -> float:
        return self.total_mrr / self.examples if self.examples else 0.0

    @property
    def ndcg_at_10(self) -> float:
        return self.total_ndcg_10 / self.examples if self.examples else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.examples if self.examples else 0.0

    @property
    def tokens_per_evidence_hit(self) -> float:
        return self.total_tokens / self.evidence_hits if self.evidence_hits else 0.0

    @property
    def avg_sources(self) -> float:
        return self.total_sources / self.examples if self.examples else 0.0

    @property
    def avg_utilization(self) -> float:
        return sum(self.utilizations) / len(self.utilizations) if self.utilizations else 0.0

    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        values = sorted(self.latencies)
        idx = int(len(values) * 0.95)
        return values[min(idx, len(values) - 1)]

    @property
    def latency_p50(self) -> float:
        if not self.latencies:
            return 0.0
        values = sorted(self.latencies)
        return values[len(values) // 2]
