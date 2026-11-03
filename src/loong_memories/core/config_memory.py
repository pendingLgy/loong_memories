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



"""记忆分区总配置：把 embedding、llm、rerank、db、vector 五类子配置聚合为一份可装载的设置。

环境变量只由 core.loader_config 解析（前缀 LOONG_MEMORIES__，双下划线分层）。
本模型主动关闭 pydantic-settings 的原生 env、.env 与 secrets 来源，避免两套装载机制并存：
否则同一份配置在 load_config(use_env=False) 与直接 MemorySettings() 下结果不一致，
通用名环境变量（如 DB、LLM）还可能被静默注入配置。
"""

from typing import Any, Tuple

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from loong_memories.core.config_db import DBSettings
from loong_memories.core.config_llm import EmbeddingSettings, LLMSettings, RerankSettings
from loong_memories.core.config_nlp import NlpSettings
from loong_memories.core.config_vector import VectorSettings


class MemorySettings(BaseSettings):
    """记忆分区：与 RagSetting 相互独立，自带 embedding、llm、rerank、db、vector 五类子配置。"""

    model_config = SettingsConfigDict(extra="forbid")

    # 总开关；为 False 时门面不构造任何记忆组件，四个分区的 enable 也不再生效。
    enable: bool = Field(default=False, description="记忆服务总开关")

    # 五类依赖：嵌入器（事实向量化）、大模型（事实抽取与摘要）、重排器（检索结果重排）、
    # 关系库（历史与消息）、向量库（记忆检索）。
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    nlp: NlpSettings = Field(default_factory=NlpSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Any,
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> Tuple[Any, ...]:
        """只保留显式入参来源。

        pydantic-settings 默认还会读取进程环境变量、.env 与 secrets，与本项目
        loader_config 的前缀式解析并存会形成双重来源。此处关闭其余来源，
        使环境变量只有一个入口，load_config 的 use_env 开关才有确定语义。
        """
        return (init_settings,)

    @model_validator(mode="after")
    def _validate_vector_dim(self) -> "MemorySettings":
        """启用向量化与向量库时，两处维度必须一致，否则写入向量库必然失败。"""
        if self.embedding.enable and self.vector.enable:
            if self.embedding.dim != self.vector.vector_size:
                raise ValueError(
                    "维度不一致：embedding.dim=" + str(self.embedding.dim)
                    + "，vector.vector_size=" + str(self.vector.vector_size)
                    + "；两者必须相同"
                )
        return self
