from typing import Any, Optional

from loong_memories.core.exceptions import ParserError


def str_to_int(value: Any) -> Optional[int]:
    """int 转换"""
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ParserError(f"parse value:{str(value)} error:{exc}") from exc


def str_to_float(value: Any) -> Optional[float]:
    """float转换"""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ParserError(f"parse value:{str(value)} error:{exc}") from exc
