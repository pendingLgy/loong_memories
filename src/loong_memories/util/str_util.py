import hashlib


def memory_hash(text: str) -> str:
    """正文指纹：用 MD5 便于与既有数据比对。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()