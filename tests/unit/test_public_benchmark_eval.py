from benchmarks.eval import (
    ExampleObservation,
    StrategyObservation,
    contains_answer,
    ndcg_at_k,
    rank_contexts_bm25,
    record_result,
    shuffle_contexts,
    write_failure_analysis,
    write_gate_report,
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


def test_gate_report_marks_failed_recall_preservation(tmp_path):
    import argparse

    results = {
        "vector": StrategyResult("Vector top-k", examples=10, evidence_hits=9, total_tokens=1000),
        "contextforge": StrategyResult(
            "Vector top-k + ContextForge", examples=10, evidence_hits=8, total_tokens=500
        ),
    }
    observations = [
        ExampleObservation(
            example_id="example-1",
            question="who wrote the example",
            answers=["answer"],
            total_relevant=1,
            strategies={
                "contextforge": StrategyObservation(
                    strategy_key="contextforge",
                    strategy="Vector top-k + ContextForge",
                    evidence_hit=True,
                    tokens=50,
                    utilization=0.5,
                    source_count=1,
                    recall_at_5_hit=True,
                    recall_at_10_hit=True,
                    mrr=1.0,
                    ndcg_at_10=1.0,
                )
            },
        )
    ]
    output = tmp_path / "gates.md"

    write_gate_report(
        results,
        observations,
        output,
        argparse.Namespace(
            n=10,
            token_budget=100,
            include_qdrant=False,
            max_recall_drop=0.01,
            min_token_reduction=0.30,
        ),
    )

    content = output.read_text()
    assert "| Vector recall preserved | FAIL | drop=0.100; max=0.010 |" in content
    assert "| Vector token reduction | PASS | reduction=50.0%; min=30.0% |" in content


def test_failure_analysis_reports_qdrant_loss(tmp_path):
    import argparse

    example = ExampleObservation(
        example_id="example-1",
        question="who wrote the example",
        answers=["answer"],
        total_relevant=1,
        strategies={
            "qdrant_hybrid": StrategyObservation(
                strategy_key="qdrant_hybrid",
                strategy="Qdrant hybrid top-k",
                evidence_hit=True,
                tokens=1000,
                utilization=0.5,
                source_count=5,
                recall_at_5_hit=True,
                recall_at_10_hit=True,
                mrr=1.0,
                ndcg_at_10=1.0,
            ),
            "qdrant_contextforge": StrategyObservation(
                strategy_key="qdrant_contextforge",
                strategy="Qdrant hybrid + ContextForge",
                evidence_hit=False,
                tokens=300,
                utilization=0.15,
                source_count=5,
                recall_at_5_hit=False,
                recall_at_10_hit=False,
                mrr=0.0,
                ndcg_at_10=0.0,
            ),
            "vector": StrategyObservation(
                strategy_key="vector",
                strategy="Vector top-k",
                evidence_hit=True,
                tokens=800,
                utilization=0.4,
                source_count=5,
                recall_at_5_hit=True,
                recall_at_10_hit=True,
                mrr=1.0,
                ndcg_at_10=1.0,
            ),
            "contextforge": StrategyObservation(
                strategy_key="contextforge",
                strategy="Vector top-k + ContextForge",
                evidence_hit=True,
                tokens=250,
                utilization=0.125,
                source_count=5,
                recall_at_5_hit=True,
                recall_at_10_hit=True,
                mrr=1.0,
                ndcg_at_10=1.0,
            ),
        },
    )
    output = tmp_path / "failures.md"

    write_failure_analysis(
        [example],
        output,
        argparse.Namespace(n=1, token_budget=2000),
    )

    content = output.read_text()
    assert "Qdrant Hybrid Evidence Lost After ContextForge" in content
    assert "`example-1`" in content
    assert "True / False" in content
