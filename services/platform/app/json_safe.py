from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID


def json_safe(value: Any) -> Any:
    """
    Recursively convert application/report objects into values that
    Python's standard json encoder and PostgreSQL JSON/JSONB can store.

    Important:
    - datetime/date -> ISO-8601 string
    - UUID -> string
    - Decimal -> float
    - Path -> string
    - numeric/boolean/string values are preserved
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(v)
            for v in value
        ]

    # Pydantic v2
    if hasattr(value, "model_dump"):
        return json_safe(
            value.model_dump()
        )

    # SQLAlchemy / dataclass-like objects should not normally reach here,
    # but objects exposing a plain dict can still be safely normalized.
    return value
