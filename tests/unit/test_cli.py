from typer.testing import CliRunner

from contextforge.cli import app

runner = CliRunner()


def test_help_exits_cleanly():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "optimize" in result.output.lower() or "Usage" in result.output


def test_optimize_requires_query():
    result = runner.invoke(app, ["optimize"])
    assert result.exit_code != 0  # missing required arg


def test_optimize_basic(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Python is a high-level programming language created by Guido van Rossum.")
    result = runner.invoke(
        app,
        ["optimize", "--query", "Python history", "--sources", str(tmp_path), "--budget", "200"],
    )
    assert result.exit_code == 0
    assert "Python" in result.output
