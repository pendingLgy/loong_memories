# 文本向量化：抽象接口、纯注册表（供用户注册自定义 embedder）与内置实现；实例化在适配器调用时进行。

from loong_memories.embedding.base import BaseEmbedder
from loong_memories.embedding.embedder_handler import EmbedderHandler
from loong_memories.embedding.openai_embedder import OpenAIEmbedder
from loong_memories.embedding.qwen_embedder import QwenEmbedder
from loong_memories.embedding.registry_embedder import (
    BUILTIN_EMBEDDERS,
    available_embedders,
    is_embedder_registered,
    register_embedder,
    register_embedder_provider,
    resolve_embedder,
)

__all__ = [
    "BaseEmbedder",
    "EmbedderHandler",
    "OpenAIEmbedder",
    "QwenEmbedder",
    "register_embedder",
    "register_embedder_provider",
    "available_embedders",
    "is_embedder_registered",
    "resolve_embedder",
    "BUILTIN_EMBEDDERS",
]
