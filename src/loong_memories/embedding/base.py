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
 嵌入模型抽象接口，隔离具体实现。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from loong_memories.core.exceptions import EmbeddingError
from loong_memories.core.config_llm import EmbeddingSettings
from loong_memories.embedding.registry_embedder import register_embedder


class BaseEmbedder(ABC):
    """嵌入模型的统一接口：声明 backend 后自动登记，配置只写 provider 名即可切换。"""

    # 配置中 embedding.provider 的取值；子类置为该名字即自动登记。
    backend: ClassVar[Optional[str]] = None
    # 未显式配置 model 时使用的模型名。
    default_model: ClassVar[str] = ""
    # 该默认模型的输出维度，供配置对齐时参考。
    default_dim: ClassVar[int] = 0
    # 未显式配置 base_url 时使用的接口地址。
    default_base_url: ClassVar[str] = ""
    # 未显式配置 api_key 时读取的环境变量名。
    api_key_env: ClassVar[str] = ""
    # 单次请求可接受的最大文本条数，0 表示不限制。
    max_batch_size: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        """子类声明 backend 后自动登记，无需额外注册调用。"""
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("backend")
        if isinstance(name, str) and name:
            # 直接登记类对象，避免本地类因 qualname 无法按路径导入。
            register_embedder(name, cls)

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings

    # ---------------- 配置解析 ----------------

    @property
    def model(self) -> str:
        """实际使用的模型名：配置优先，缺省回落到 default_model。"""
        return self._settings.model or self.default_model

    @property
    def base_url(self) -> str:
        """实际使用的接口地址：配置优先，缺省回落到 default_base_url。"""
        return self._settings.base_url or self.default_base_url

    @property
    def dim(self) -> int:
        """期望的输出维度：由 embedding.dim 声明，供向量化结果校验。"""
        return self._settings.dim

    @property
    def api_key(self) -> str:
        """API Key：配置优先，其次读取 embedder 约定的环境变量。"""
        key = self._settings.api_key or os.environ.get(self.api_key_env, "")
        if not key:
            raise EmbeddingError(
                "缺少 " + str(
                    self.backend) + " 的 API Key：请设置环境变量 " + self.api_key_env + "，或配置 embedding.api_key"
            )
        return key

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """把一批文本编码为向量，顺序与输入一致。"""
