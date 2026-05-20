def test_public_imports():
    from contextforge import ContextEngine, ContextWindow, RetentionEvaluator, Source, SourceType
    assert all([ContextEngine, Source, SourceType, ContextWindow, RetentionEvaluator])


def test_three_line_usage():
    from contextforge import ContextEngine, Source
    engine = ContextEngine(token_budget=200, top_k=5, top_n=2)
    window = engine.build(
        query="Python programming",
        sources=[Source(content="Python is a language.", path="doc.txt")]
    )
    assert isinstance(window.render(), str)
