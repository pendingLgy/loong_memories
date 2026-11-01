from datetime import datetime, timezone


def ensure_datetime(val):
    """确保输入是 datetime 对象，如果是字符串则自动解析，为空则给当前 UTC 时间"""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except ValueError:
            pass
    return None


def now_datetime():
    """当前utc时间"""
    return datetime.now(timezone.utc)

def now_iso() -> str:
    """当前时间：统一 UTC ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()

def datetime_to_isoformat(val):
    """当前时间：统一 UTC ISO 8601 字符串。"""

    if isinstance(val, datetime):
        return val.isoformat()
    else:
        return val
