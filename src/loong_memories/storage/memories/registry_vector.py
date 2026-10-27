# 向量库后端注册表：支持注册自定义实现，只负责注册与解析，不做实例化。
#
# 注册方式二选一：继承 VectorStoreBase 并声明 backend（自动注册），或调用 register_vector_store 显式登记。
# 注册值可为类对象，也可为点分路径；实例化在 vector_engine 中进行，本模块不持有任何实例。

from __future__ import annotations

from typing import Dict, List, Any

from loong_memories.registry import Ref, Registry

# 内置后端名称到类路径的映射，按需惰性导入。
BUILTIN_STORES: Dict[str, str] = {
    "milvus": "loong_memories.storage.memories.milvus_store.MilvusVectorStore",
    "milvus_async": "loong_memories.storage.memories.milvus_store.AsyncMilvusVectorStore",
    "pg": "loong_memories.storage.memories.pg_store.PGVectorStore",
    "pg_async": "loong_memories.storage.memories.pg_store.AsyncPGVectorStore",
}

_REGISTRY = Registry("向量库后端", BUILTIN_STORES)


def register_vector_store(name: str, ref: Ref) -> None:
    "登记向量库后端：ref 可为类对象，也可为点分路径字符串。"
    _REGISTRY.register(name, ref)


def register_vector_store_provider(name: str):
    "装饰器形式登记向量库后端。"
    return _REGISTRY.decorator(name)


def is_vector_store_registered(name: str) -> bool:
    "判断向量库后端是否已登记。"
    return name in _REGISTRY


def available_vector_stores() -> List[str]:
    "返回已登记的后端名，按字典序排列。"
    return _REGISTRY.names()


def resolve_vector_store(name: str) -> Any:
    "按后端名取得实现类，路径登记的在此惰性导入。"
    return _REGISTRY.resolve(name)
