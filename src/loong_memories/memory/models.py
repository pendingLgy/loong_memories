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
 记忆层数据对象：对外只暴露 pydantic 模型，避免裸字典在层间传递。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryItem(BaseModel):
    """一条记忆：正文、分数与元数据。"""

    model_config = ConfigDict(extra="ignore")

    id: int
    memory: str
    # 检索场景下的融合分数；写入与读取场景为 None。
    score: Optional[float] = None
    hash: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # 作用域标识，来源为显式参数，不接受 metadata 覆盖。
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    actor_id: Optional[str] = None
    role: Optional[str] = None
    # 业务自定义元数据，不含上述核心键。
    metadata: Optional[Dict[str, Any]] = None


class AddResult(BaseModel):
    """写入结果：本次新增的记忆条目。"""

    model_config = ConfigDict(extra="ignore")

    results: List[MemoryItem] = Field(default_factory=list)


class SearchResult(BaseModel):
    """检索结果：按融合分数降序排列的记忆条目。"""

    model_config = ConfigDict(extra="ignore")

    results: List[MemoryItem] = Field(default_factory=list)


class UpdateResult(BaseModel):
    """更新结果：更新后的记忆条目。"""

    model_config = ConfigDict(extra="ignore")

    result: Optional[MemoryItem] = None


class DeleteResult(BaseModel):
    """删除结果：被删除记忆的 id。"""

    model_config = ConfigDict(extra="ignore")

    id: int
