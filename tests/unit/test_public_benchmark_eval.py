from benchmarks.eval import contains_answer, record_result
from benchmarks.schemas import StrategyResult
from contextforge.budget import TokenCounter


def test_contains_answer_normalizes_whitespace_and_case():
    assert contains_answer("The answer is Louis   de Broglie.", ["louis de broglie"])


def test_record_result_tracks_evidence_recall_and_utilization():
    result = StrategyResult("demo")
    record_result(
        result,
        rendered_context="Louis de Broglie proposed it.",
        source_count=2,
        token_budget=100,
        answers=["Louis de Broglie"],
        counter=TokenCounter(),
        latency_s=0.25,
    )

    assert result.examples == 1
    assert result.evidence_hits == 1
    assert result.evidence_recall == 1.0
    assert result.avg_sources == 2.0
    assert result.latency_p95 == 0.25
