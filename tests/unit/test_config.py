from contextforge.core.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.provider == "anthropic"
    assert s.token_budget == 8000
    assert s.top_k == 20
    assert s.top_n == 5


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("CONTEXTFORGE_TOKEN_BUDGET", "4000")
    s = Settings()
    assert s.token_budget == 4000
