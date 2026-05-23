from benchmarks.eval import (
    contains_answer,
    ndcg_at_k,
    rank_contexts_bm25,
    record_result,
    shuffle_contexts,
)
from benchmarks.schemas import BenchmarkExample, StrategyResult
from contextforge.budget import TokenCounter


def test_contains_answer_normalizes_whitespace_and_case():
    assert contains_answer("The answer is Louis   de Broglie.", ["louis de broglie"])


def test_shuffle_contexts_is_deterministic_per_example():
    example = BenchmarkExample(
        id="nq-1",
        question="who proposed wave particle duality",
        answers=["Louis de Broglie"],
        contexts=["a", "b", "c", "d", "e"],
    )

    first = shuffle_contexts(example, seed=13)
    second = shuffle_contexts(example, seed=13)

    assert first.contexts == second.contexts
    assert first.contexts != example.contexts
    assert first.metadata["candidate_order"] == "shuffled"


def test_bm25_ranks_exact_terms_first():
    contexts = [
        "General physics history.",
        "Louis de Broglie proposed matter waves for electrons.",
        "Basketball awards and all star games.",
    ]

    ranked = rank_contexts_bm25("who proposed electron matter waves", contexts)

    assert ranked[0] == "Louis de Broglie proposed matter waves for electrons."


def test_ndcg_uses_full_relevant_count_for_ideal_dcg():
    assert ndcg_at_k([0], total_relevant=1, k=10) == 1.0
    assert 0 < ndcg_at_k([2], total_relevant=1, k=10) < 1.0


def test_record_result_tracks_ir_metrics_and_utilization():
    result = StrategyResult("demo")
    record_result(
        result,
        rendered_context="Louis de Broglie proposed it.",
        ranked_contexts=[
            "Distractor context.",
            "Louis de Broglie proposed it.",
            "Another distractor.",
        ],
        total_relevant=1,
        source_count=2,
        token_budget=100,
        answers=["Louis de Broglie"],
        counter=TokenCounter(),
        latency_s=0.25,
    )

    assert result.examples == 1
    assert result.evidence_hits == 1
    assert result.evidence_recall == 1.0
    assert result.recall_at_5 == 1.0
    assert result.recall_at_10 == 1.0
    assert result.mrr == 0.5
    assert result.ndcg_at_10 > 0
    assert result.avg_sources == 2.0
    assert result.tokens_per_evidence_hit == result.avg_tokens
    assert result.latency_p50 == 0.25
    assert result.latency_p95 == 0.25
