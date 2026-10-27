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



from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from loong_memories.core.config_db import DBSettings
from loong_memories.core.config_llm import EmbeddingSettings, LLMSettings
from loong_memories.core.config_vector import VectorSettings


class MemorySettings(BaseSettings):
    """记忆分区：与 RagSetting 相互独立，自带 embedding、llm、db、vector 四类子配置。"""

    model_config = SettingsConfigDict(
        # 前缀以单下划线结尾、嵌套又以双下划线分隔，原生解析会把 LOONG_MEMORIES__DB__BACKEND
        # 切成 _DB 与 BACKEND；故关闭原生 env 解析与 .env 文件，改由
        # settings_customise_sources 挂载 _MemoryEnvSource，与 load_env 共用同一套解析。
        # 未知字段仍直接报错，避免拼写错误被静默忽略。
        extra="forbid",
    )

    # 四类依赖：嵌入器（事实向量化）、大模型（事实抽取）、关系库（历史与消息）、向量库（记忆检索）。
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)