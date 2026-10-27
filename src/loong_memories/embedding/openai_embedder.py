# OpenAI 嵌入实现；Qwen 等 OpenAI 兼容服务复用同一套请求格式。
#
# 使用官方 openai SDK，提升维护性并简化请求与异常处理。

from __future__ import annotations

from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI

from loong_memories.core.exceptions import EmbeddingError
from loong_memories.embedding.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """调用 OpenAI 的 embeddings 接口。"""

    backend = "openai"
    default_model = "text-embedding-3-small"
    default_dim = 1536
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # 初始化 SDK 客户端，传递 base_url 与 api_key
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """惰性初始化 OpenAI 客户端"""
        if self._client is None:
            timeout = self._settings.timeout
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
            )
        return self._client

    def encode(self, texts: List[str]) -> List[List[float]]:
        """请求一次 embeddings，返回与输入同序的向量列表。"""
        kwargs = self._build_request_kwargs(texts)
        try:
            response = self.client.embeddings.create(**kwargs)
            # 使用 response.model_dump() 转换为字典格式，保持与原 _parse 流程相兼容
            return self._parse(response.model_dump())
        except openai.APIConnectionError as exc:
            raise EmbeddingError("嵌入请求无法连接：" + str(exc)) from exc
        except openai.APIStatusError as exc:
            raise EmbeddingError(
                f"嵌入请求失败（HTTP {exc.status_code}）：{exc.message}"
            ) from exc
        except openai.OpenAIError as exc:
            raise EmbeddingError("嵌入请求发生异常：" + str(exc)) from exc

    def _build_request_kwargs(self, texts: List[str]) -> Dict[str, Any]:
        """构建 SDK 传递给 client.embeddings.create 的 kwargs 参数字典。"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }

        # 仅在显式配置了维度、且与默认维度不同时才传 dimensions，兼容不支持该参数的模型。
        configured = self._settings.dim
        if configured and configured != self.default_dim:
            kwargs["dimensions"] = configured

        return kwargs

    def _parse(self, data: Dict[str, Any]) -> List[List[float]]:
        """按 index 排序还原响应顺序，保证与输入一一对应。"""
        items = data.get("data") or []
        if not items:
            raise EmbeddingError("嵌入响应缺少 data 字段：" + str(list(data.keys())))
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        return [[float(value) for value in item["embedding"]] for item in ordered]
