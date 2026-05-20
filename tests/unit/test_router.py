from contextforge.models.source import SourceType
from contextforge.router import ContentTypeRouter


def test_detects_prose():
    router = ContentTypeRouter()
    text = "The quick brown fox jumps over the lazy dog. It was a sunny morning."
    assert router.detect(text) == SourceType.PROSE


def test_detects_python():
    router = ContentTypeRouter()
    assert router.detect("def calculate(x: int) -> int:\n    return x * 2") == SourceType.CODE


def test_detects_code_block():
    router = ContentTypeRouter()
    assert router.detect("Example:\n```python\nimport os\n```") == SourceType.CODE


def test_detects_json():
    router = ContentTypeRouter()
    assert router.detect('{"name": "contextforge", "version": "0.1.0"}') == SourceType.STRUCTURED


def test_detects_yaml():
    router = ContentTypeRouter()
    assert router.detect("name: contextforge\nversion: 0.1.0\ndeps:\n  - pydantic") == SourceType.STRUCTURED


def test_detects_javascript():
    router = ContentTypeRouter()
    assert router.detect("function greet(name) {\n  return `Hello, ${name}`;\n}") == SourceType.CODE
