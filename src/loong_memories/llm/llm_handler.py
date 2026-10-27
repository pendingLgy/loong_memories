# Copyright [2026] [longgyx]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
大模型处理器：provider 由配置选择，实例化在调用时进行；模型句柄可注入以便测试。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from loong_memories.core.config_llm import LLMSettings
from loong_memories.llm.registry_llm import resolve_llm_provider

logger = logging.getLogger(__name__)


class LLMHandler:
    """大模型处理器：每次调用按配置解析并实例化 provider，注入的 model 优先且绕过配置。"""

    def __init__(self, settings: LLMSettings, model: Optional[Any] = None) -> None:
        self._settings = settings
        # 注入的 model 只需实现 chat(messages, **overrides)，便于测试与自定义。
        self._model = model
        self._provider: Optional[Any] = None

    @property
    def provider(self) -> Any:
        """实际使用的调用对象：注入的 model 优先，否则按配置装配 provider。"""
        return self._target()

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> str:
        """单轮调用大模型，返回回复文本。"""
        return self._target().chat(messages, **overrides)

    def complete(self, prompt: str, system: str = "", **overrides: Any) -> str:
        """以单条用户消息完成一次生成，可选系统提示。"""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)

    # ---------------- 内部实现 ----------------

    def _target(self) -> Any:
        # 注册表只登记实现类，实例化在这里完成并按需缓存，避免重复构造。
        if self._model is not None:
            return self._model
        if self._provider is None:
            name = self._settings.provider
            provider_cls = resolve_llm_provider(name)
            self._provider = provider_cls(self._settings)
            logger.info(
                "LLM provider 已装配 provider=%s model=%s",
                name,
                getattr(self._provider, "model", ""),
            )
        return self._provider

