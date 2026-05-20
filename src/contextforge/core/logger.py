from __future__ import annotations

import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    return structlog.get_logger(name)
