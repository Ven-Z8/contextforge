from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    STRUCTURED = "structured"


@dataclass
class Source:
    content: str
    source_type: SourceType = SourceType.PROSE
    path: str | None = None
    source_id: str | None = None
    metadata: dict = field(default_factory=dict)
