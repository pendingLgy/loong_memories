# OpenAI 大模型实现；qwen、deepseek 等 OpenAI 兼容服务复用同一套请求格式。
#
# 使用官方 openai SDK，提升可维护性并简化请求与异常处理。

from __future__ import annotations

from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI
from loong_memories.core.exceptions import LLMError
from loong_memories.llm.base import BaseLLMProvider


class OpenAILLMProvider(BaseLLMProvider):
    """调用 OpenAI 的 chat completions 接口。"""

    backend = "openai"
    default_model = "gpt-4o-mini"
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """惰性初始化 OpenAI 客户端（api_key、base_url、timeout）。"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self._settings.timeout,
            )
        return self._client

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> str:
        """请求一次 chat completions，返回首条回复文本。"""
        kwargs = self._build_request_kwargs(messages, overrides)
        try:
            response = self.client.chat.completions.create(**kwargs)
            # 用 model_dump 转字典，保持与原 _parse 流程兼容。
            return self._parse(response.model_dump())
        except openai.APIConnectionError as exc:
            raise LLMError("大模型请求无法连接：" + str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"大模型请求失败（HTTP {exc.status_code}）：{exc.message}") from exc
        except openai.OpenAIError as exc:
            raise LLMError("大模型请求发生异常：" + str(exc)) from exc

    # ---------------- 请求构造（子类可覆盖以适配不同厂商） ----------------

    def _build_request_kwargs(self, messages: List[Dict[str, str]], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """构造 SDK 传给 chat.completions.create 的参数字典。"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
        }
        # max_tokens 为 0 时不传，交由服务端决定上限。
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        kwargs.update(overrides)
        return kwargs

    def _parse(self, data: Dict[str, Any]) -> str:
        """从响应中取出首条 choices 的 message.content。"""
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("大模型响应缺少 choices 字段：" + str(list(data.keys())))
        message = choices[0].get("message") or {}
        return message.get("content") or ""
