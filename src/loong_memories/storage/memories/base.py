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
 向量库抽象接口：屏蔽具体实现（Milvus、PG 等）。

 约定：search 返回的 score 越大越相似（建议范围 [0, 1]）。
 采用距离度量的实现需先转相似度：
   COSINE 距离：score = max(0.0, 1.0 - distance)
   L2 距离：score = 1.0 / (1.0 + distance)
   IP 内积：score 即内积值（越大越相似）
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from loong_memories.storage.memories.registry_vector import register_vector_store

# 支持的向量度量类型。
MetricType = Literal["COSINE", "L2", "IP"]


class VectorStoreResult(BaseModel):
    """一次检索或读取得到的结果：id、相似度与元数据。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    tenant_id: Optional[int] = None
    user_id: Optional[int] = None
    text: Optional[str] = None
    text_lemma: Optional[str] = None
    updated_at_tz: Optional[str] = None
    created_at_tz: Optional[str] = None

    # 相似度分数，越大越相似；list、get 等无排序语境下为 None。
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def to_vector_store_result(cls, **kwargs) -> VectorStoreResult:
        id = int(kwargs.get("id")) if kwargs.get("id") else None
        tenant_id = int(kwargs.get("tenant_id")) if kwargs.get("tenant_id") else None
        user_id = int(kwargs.get("user_id")) if kwargs.get("user_id") else None
        score = float(kwargs.get("score")) if kwargs.get("score") is not None else None

        return VectorStoreResult(
            id=id,
            tenant_id=tenant_id,
            user_id=user_id,
            text=kwargs.get("text"),
            text_lemma=kwargs.get("text_lemma"),
            updated_at_tz=kwargs.get("updated_at_tz"),
            created_at_tz=kwargs.get("created_at_tz"),
            score=score,
            metadata=kwargs.get("metadata")
        )

class VectorStoreBase(ABC):
    """向量库统一接口：声明 backend 后自动登记，配置改后端名即可切换。"""

    # 配置中的向量库后端名；子类置为该名字即自动登记。
    backend: ClassVar[Optional[str]] = None
    # 距离度量，子类可覆盖默认值。
    metric_type: ClassVar[MetricType] = "COSINE"

    def __init_subclass__(cls, **kwargs: object) -> None:
        """子类声明 backend 后自动登记，无需额外注册调用。"""
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("backend")
        if isinstance(name, str) and name:
            # 直接登记类对象，避免本地类因 qualname 无法按路径导入。
            register_vector_store(name, cls)

    @abstractmethod
    def start(self) -> Any:
        """启动连接"""

    @abstractmethod
    def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        """新建集合（PG 中即建表）。"""

    @abstractmethod
    def insert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[int] ) -> None:
        """批量写入向量及其元数据；ids 缺省时由实现自行生成。"""

    @abstractmethod
    def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        """按相似度检索，返回按 score 降序的结果。"""

    @abstractmethod
    def delete(self, id: int) -> None:
        """按 id 删除单条。"""

    @abstractmethod
    def update(self, id: int, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        """按 id 更新向量与元数据（可只更新其一）。"""

    @abstractmethod
    def get(self, id: int) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""

    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> Optional[List[VectorStoreResult]]:
        """列出集合内容（外层列表按既有约定包裹一层）。"""

    def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        """关键字（BM25 或全文）检索；不支持的实现返回 None。"""
        return None

    def _ensure_datetime(self, val):
        """确保输入是 datetime 对象，如果是字符串则自动解析，为空则给当前 UTC 时间"""
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace('Z', '+00:00'))
            except ValueError:
                pass
        return None

    def _build_metadata(self, payloads:Dict[str, Any]):
        keys_to_remove = {"text", "text_lemma", "updated_at_tz", "created_at_tz"}

        metadata = {
            k: v for k, v in (payloads or {}).items()
            if k not in keys_to_remove
        }

        return metadata