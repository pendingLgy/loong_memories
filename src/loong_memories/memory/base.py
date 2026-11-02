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
 记忆层抽象契约：四条同步流程，与 storage 等子包同构。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loong_memories.memory.models import AddResult, DeleteResult, SearchResult, UpdateResult, GetResult, PageQueryResult


class MemoryBase(ABC):
    """记忆层统一接口：写入、检索、更新、删除。"""

    @abstractmethod
    def get(
            self,
            memory_id: Any,
            tenant_id: int,
            user_id: Optional[int] = None,
    ) -> GetResult:
        """获取记忆"""

    @abstractmethod
    def page_query(
        self,
        tenant_id: int,
        user_id: Optional[int] = None,
        page_num: int = 0,
        page_size: int = 10,
        *,
        filter: Optional[Dict[str, Any]] = None
    ) -> PageQueryResult:
        """分页查询记忆"""

    @abstractmethod
    def add(
            self,
            messages: Any,
            tenant_id: int,
            user_id: int | None = None,
            *,
            agent_id: Optional[Any] = None,
            metadata: Dict[str, Any] = None,
    ) -> AddResult:
        """把对话或文本写入记忆，返回本次新增的条目。"""

    @abstractmethod
    def search(
        self,
        query: str,
        tenant_id: int,
        user_id: Optional[int] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        rerank: bool = False,
        explain: bool = False,
    ) -> SearchResult:
        """语义与关键词双路召回，融合排序后返回带分数的条目。"""

    @abstractmethod
    def update(
            self,
            memory_id: int,
            tenant_id: int,
            user_id: int | None = None,
            *,
            text: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> UpdateResult:
        """更新正文或元数据；正文变化时重算向量。"""

    @abstractmethod
    def delete(self,
               memory_id: int,
               tenant_id: int,
               user_id: int | None = None) -> DeleteResult:
        """删除记忆：向量库物理删除，历史表留审计。"""
