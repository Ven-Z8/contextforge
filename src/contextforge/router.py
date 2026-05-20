from __future__ import annotations

import re

from contextforge.models.source import SourceType

_CODE_PATTERNS = [
    r"```",
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\bfunction\s+\w+\s*\(",
    r"\bconst\s+\w+\s*=",
    r"\bvar\s+\w+\s*=",
    r"\blet\s+\w+\s*=",
    r"\breturn\s+.+;",
    r"#include\s*<",
    r"\bpublic\s+(?:static\s+)?\w+\s+\w+\s*\(",
]

_STRUCTURED_PATTERNS = [
    r'^\s*[\[{"]',
    r"^\w[\w\s]*:\s+\S",
    r"<\w+>.*</\w+>",
]


class ContentTypeRouter:
    """Detects whether content is prose, code, or structured data.

    IMPORTANT: Code and structured data are never lossy-compressed.
    Safe operations (whole-function selection, whole-object selection) are allowed
    but sentence-level compression is not applied to these types.
    """

    def detect(self, text: str) -> SourceType:
        for pattern in _CODE_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                return SourceType.CODE
        for pattern in _STRUCTURED_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                return SourceType.STRUCTURED
        return SourceType.PROSE
