# 嵌入模型注册表：支持用户注册自定义 embedder，只负责注册与解析，不做实例化。
#
# 注册方式二选一：继承 BaseEmbedder 并声明 backend（自动注册），或调用 register_embedder 显式登记。
# 注册值可为类对象，也可为点分路径（首次解析时惰性导入）。
# 实例化在 EmbedderHandler 调用时进行，本模块不持有任何实例。

from __future__ import annotations

from typing import Dict, List, Type

from loong_memories.embedding import BaseEmbedder
from loong_memories.registry import Ref, Registry

# 内置 embedder 名称到类路径的映射，按需惰性导入。
BUILTIN_EMBEDDERS: Dict[str, str] = {
    "openai": "loong_memories.embedding.openai_embedder.OpenAIEmbedder",
    "qwen": "loong_memories.embedding.qwen_embedder.QwenEmbedder",
}

_REGISTRY = Registry("嵌入模型", BUILTIN_EMBEDDERS)


def register_embedder(name: str, ref: Ref) -> None:
    """登记嵌入模型：ref 可为类对象，也可为点分路径字符串。"""
    _REGISTRY.register(name, ref)


def register_embedder_provider(name: str):
    """装饰器形式登记嵌入模型。"""
    return _REGISTRY.decorator(name)


def is_embedder_registered(name: str) -> bool:
    """判断嵌入模型是否已登记。"""
    return name in _REGISTRY


def available_embedders() -> List[str]:
    """返回已登记的 embedder 名，按字典序排列。"""
    return _REGISTRY.names()


def resolve_embedder(name: str) -> Type[BaseEmbedder]:
    """按 embedder 名取得实现类，路径登记的在此时惰性导入。"""
    return _REGISTRY.resolve(name)
