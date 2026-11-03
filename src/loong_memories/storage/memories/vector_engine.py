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
 向量引擎：按后端名构造向量库，并统一封装向量数据的读写调用。

 与 db_engine.build_database 对称：构造与调用细节集中在引擎内，注册表只登记实现类。
 后端实现按需惰性导入，避免与本模块形成导入环。
"""


from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from loong_memories.core.exceptions import ConfigError, StorageError
from loong_memories.storage.memories.base import VectorStoreBase, VectorStoreResult
from loong_memories.storage.memories.registry_vector import available_vector_stores

logger = logging.getLogger(__name__)


# 默认后端、集合名与向量维度。
DEFAULT_BACKEND = 'milvus'
DEFAULT_COLLECTION = ''
DEFAULT_VECTOR_SIZE = 1024

# 默认检索条数。
DEFAULT_TOP_K = 5


def build_vector_store(backend: str = DEFAULT_BACKEND, **kwargs: Any) -> VectorStoreBase:
    """按后端名构造向量库实例；未登记的后端抛 ConfigError。"""
    if backend == 'milvus':
        from loong_memories.storage.memories.milvus_store import MilvusVectorStore
        return MilvusVectorStore(**kwargs)
    if backend == 'pg':
        from loong_memories.storage.memories.pg_store import PGVectorStore
        return PGVectorStore(**kwargs)
    raise ConfigError(
        "未登记的向量库后端：" + str(backend) + "；可用：" + ", ".join(available_vector_stores())
    )


class VectorEngine:
    """向量数据调用入口：持有向量库实现，统一记录日志并校验入参。"""

    def __init__(
        self,
        store_base: VectorStoreBase,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:

        store_backend = store_base.backend
        # if "async" in store_backend:
        #     raise StorageError("not support async engine")

        self._store_base = store_base
        self.default_top_k = default_top_k

    def start(self):
        """启动数据库连接"""
        self._store_base.start()

    # ---------------- 属性 ----------------

    @property
    def store(self) -> VectorStoreBase:
        """底层向量库实现，便于使用未封装的接口。"""
        return self._store_base

    @property
    def backend(self) -> Optional[str]:
        """后端名，取自实现类的 backend 声明。"""
        return getattr(self._store_base, "backend", None)

    @property
    def collection_name(self) -> str:
        """当前集合名。"""
        return str(getattr(self._store_base, "collection_name", ""))

    @property
    def vector_size(self) -> int:
        """集合向量维度，未声明时为 0。"""
        return int(getattr(self._store_base, "vector_size", 0) or 0)


    # ---------------- 写入 ----------------

    def insert(
        self,
        vectors: List[List[float]],
        payloads:List[Dict[str, Any]],
        ids: List[int]
    ) -> int:
        """写入向量及其元数据，返回写入条数。"""
        if not vectors:
            return 0
        self._check_batch(vectors, payloads, ids)
        self._store_base.insert(vectors, payloads=payloads, ids=ids)
        self._log("向量写入完成", count=len(vectors), collection=self.collection_name)
        return len(vectors)


    def update(
        self,
        vector_id: int,
        tenant_id: int,
        user_id: Optional[int] = None,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """按 id 更新向量或元数据。"""
        if vector is not None:
            self._check_vector(vector)
        self._store_base.update(vector_id, tenant_id, user_id=user_id, vector=vector, payload=payload)
        self._log("向量更新完成", id=vector_id, collection=self.collection_name)


    def delete(self, vector_id: int, tenant_id: int, user_id: Optional[int] = None) -> None:
        """按 id 删除单条。"""
        self._store_base.delete(vector_id, tenant_id, user_id=user_id)
        self._log("向量删除完成", id=vector_id, collection=self.collection_name)


    # ---------------- 检索 ----------------

    def search(
        self,
        query: str,
        vectors: List[float],
        tenant_id: int,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreResult]:
        """按相似度检索，返回按 score 降序的结果；top_k 缺省时用 default_top_k。"""
        self._check_vector(vectors)
        limit = self.default_top_k if top_k is None else top_k
        results = self._store_base.search(query, vectors, tenant_id, user_id=user_id,
                                    top_k=limit, filter=filter)
        self._log("向量检索完成", top_k=limit, hits=len(results), collection=self.collection_name)
        return results


    def keyword_search(
        self,
        query: str,
        tenant_id: int,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[VectorStoreResult]]:
        """关键词检索；实现不支持时返回 None。"""
        limit = self.default_top_k if top_k is None else top_k
        results = self._store_base.keyword_search(query, tenant_id, user_id=user_id,
                                            top_k=limit, filter=filter)
        hits = 0 if results is None else len(results)
        self._log("关键词检索完成", top_k=limit, hits=hits, collection=self.collection_name)
        return results


    def get(self, vector_id: int, tenant_id: int, user_id: Optional[int] = None) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""
        return self._store_base.get(vector_id, tenant_id, user_id=user_id)

    def list(self, tenant_id: int, user_id: Optional[int] = None,
             filter: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[VectorStoreResult]:
        return self._store_base.list(tenant_id, user_id=user_id, filter=filter, top_k=top_k) or []


    # ---------------- 内部实现 ----------------

    def _check_vector(self, vector: List[float]) -> None:
        """校验向量维度与集合声明一致。"""
        expected = self.vector_size
        if expected and len(vector) != expected:
            raise StorageError("向量维度不一致：期望 " + str(expected) + "，实际 " + str(len(vector)))


    def _check_batch(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]],
        ids: Optional[List[int]],
    ) -> None:
        """校验批量入参：维度一致，且 payloads 与 ids 数量对齐。"""
        for vector in vectors:
            self._check_vector(vector)
        if payloads is not None and len(payloads) != len(vectors):
            raise StorageError("payloads 与 vectors 数量不一致：" + str(len(payloads)) + " != " + str(len(vectors)))
        if ids is not None and len(ids) != len(vectors):
            raise StorageError("ids 与 vectors 数量不一致：" + str(len(ids)) + " != " + str(len(vectors)))


    def _log(self, message: str, **fields: Any) -> None:
        """记录调用日志：字段以 key=value 追加在消息末尾。"""
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info(f"{message} {detail}" if detail else message)


class AsyncVectorEngine(VectorEngine):
    """向量数据调用入口：持有向量库实现，统一记录日志并校验入参。"""

    def __init__(self, store_base: VectorStoreBase, default_top_k: int = DEFAULT_TOP_K) -> None:

        super().__init__(store_base, default_top_k)
        store_base_backend = store_base.backend
        if "async" not in store_base_backend:
            raise StorageError("not support sync engine")

        self._store_base = store_base
        self.default_top_k = default_top_k

    async def start(self):
        """启动数据库连接"""
        await self._store_base.start()

    # ---------------- 写入 ----------------
    async def insert(
            self,
            vectors: List[List[float]],
            payloads: List[Dict[str, Any]],
            ids: List[int]
    ) -> int:
        """写入向量及其元数据，返回写入条数。"""
        if not vectors:
            return 0
        self._check_batch(vectors, payloads, ids)
        await self._store_base.insert(vectors, payloads=payloads, ids=ids)
        self._log("向量写入完成", count=len(vectors), collection=self.collection_name)
        return len(vectors)


    async def update(
            self,
            vector_id: int,
            tenant_id: int,
            user_id: Optional[int] = None,
            vector: Optional[List[float]] = None,
            payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """按 id 更新向量或元数据。"""
        if vector is not None:
            self._check_vector(vector)
        await self._store_base.update(vector_id, tenant_id, user_id=user_id, vector=vector, payload=payload)
        self._log("向量更新完成", id=vector_id, collection=self.collection_name)


    async def delete(self, vector_id: int, tenant_id: int, user_id: Optional[int] = None) -> None:
        """按 id 删除单条。"""
        await self._store_base.delete(vector_id, tenant_id, user_id=user_id)
        self._log("向量删除完成", id=vector_id, collection=self.collection_name)


    # ---------------- 检索 ----------------

    async def search(
        self,
        query: str,
        vectors: List[float],
        tenant_id: int,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreResult]:
        """按相似度检索，返回按 score 降序的结果；top_k 缺省时用 default_top_k。"""
        self._check_vector(vectors)
        limit = self.default_top_k if top_k is None else top_k
        results = await self._store_base.search(query, vectors, tenant_id, user_id=user_id,
                                          top_k=limit, filter=filter)
        self._log("向量检索完成", top_k=limit, hits=len(results), collection=self.collection_name)
        return results


    async def keyword_search(
        self,
        query: str,
        tenant_id: int,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[VectorStoreResult]]:
        """关键词检索；实现不支持时返回 None。"""
        limit = self.default_top_k if top_k is None else top_k
        results = await self._store_base.keyword_search(query, tenant_id, user_id=user_id,
                                                  top_k=limit, filter=filter)
        hits = 0 if results is None else len(results)
        self._log("关键词检索完成", top_k=limit, hits=hits, collection=self.collection_name)
        return results


    async def get(self, vector_id: int, tenant_id: int, user_id: Optional[int] = None) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""
        return await self._store_base.get(vector_id, tenant_id, user_id=user_id)


    async def list(self,
                   tenant_id: int,
                   user_id: Optional[int] = None,
                   filter: Optional[Dict[str, Any]] = None,
                   top_k: Optional[int] = None) -> Optional[List[VectorStoreResult]]:
        return await self._store_base.list(tenant_id, user_id=user_id, filter=filter, top_k=top_k) or []


def create_vector_engine(
    backend: str = DEFAULT_BACKEND,
    default_top_k: int = DEFAULT_TOP_K,
    **kwargs: Any,
) -> VectorEngine:
    """构造向量库并包成引擎，一次得到可直接调用的入口。"""
    store_base = build_vector_store(backend, **kwargs)
    return VectorEngine(store_base, default_top_k=default_top_k)
