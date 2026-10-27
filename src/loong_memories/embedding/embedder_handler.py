# 向量化处理器：embedder 由配置选择，实例化在调用时进行；模型句柄可注入以便测试。

from __future__ import annotations

import logging
from typing import Any, List, Optional

from loong_memories.core.config_llm import EmbeddingSettings
from loong_memories.core.exceptions import EmbeddingError
from loong_memories.embedding.registry_embedder import resolve_embedder

logger = logging.getLogger(__name__)


class EmbedderHandler:
    """向量化处理器：按配置解析并实例化 embedder，注入的 model 优先且绕过配置。"""

    def __init__(self, settings: EmbeddingSettings, model: Optional[Any] = None) -> None:
        self._settings = settings
        # 注入的 model 只需实现 encode(texts)，可选提供 max_batch_size，便于测试与自定义。
        self._model = model
        self._provider: Optional[Any] = None

    @property
    def provider(self) -> Any:
        """实际使用的编码对象：注入的 model 优先，否则按配置装配 embedder。"""
        return self._target()

    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为稠密向量，返回顺序与输入一致。"""
        if not texts:
            return []
        target = self._target()
        step = self._batch_size(target)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), step):
            vectors.extend(self._encode_batch(target, texts[start:start + step]))
        self._validate_dim(vectors)
        return vectors

    # ---------------- 内部实现 ----------------

    def _target(self) -> Any:
        if self._model is not None:
            return self._model
        if self._provider is None:
            name = self._settings.provider
            embedder_cls = resolve_embedder(name)
            self._provider = embedder_cls(self._settings)

            logger.info(
                "Embedder 已装配 provider=%s model=%s dim=%s",
                name,
                getattr(self._provider, "model", ""),
                getattr(self._provider, "dim", 0),
            )
        return self._provider

    def _batch_size(self, target: Any) -> int:
        """批次大小：取配置值与 embedder 单次上限的较小者，上限为 0 表示不限制。"""
        size = self._settings.batch_size
        limit = int(getattr(target, "max_batch_size", 0) or 0)
        return min(size, limit) if limit > 0 else size

    def _encode_batch(self, target: Any, batch: List[str]) -> List[List[float]]:
        """编码单批文本；是否归一化由 embedder 决定，此处仅透传结果。"""
        result = target.encode(batch)
        return [list(map(float, row)) for row in result]

    def _validate_dim(self, vectors: List[List[float]]) -> None:
        """校验向量维度与 embedding.dim 一致，避免下游按错误维度使用。"""
        expected = self._settings.dim
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    "向量维度不一致：期望 " + str(expected) + "，实际 " + str(len(vector))
                )
