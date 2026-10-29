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
 存储薄封装：统一 id 转换、payload 布局与批量回退策略。

 记忆层不直接接触方言实现，只依赖 DBEngine 与 VectorEngine 的抽象方法。
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loong_memories.core.exceptions import StorageError
from loong_memories.memory.models import MemoryItem
from loong_memories.nlp.lemmatization import lemmatize_zh_for_bm25
from loong_memories.storage.memories.base import VectorStoreResult
from loong_memories.storage.memories.vector_engine import VectorEngine, AsyncVectorEngine
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine
from loong_memories.storage.message.schema import HistoryRecord, Message
from loong_memories.util.scoring import get_bm25_params, normalize_bm25, score_and_rank
from loong_memories.util.snowflake import next_id

logger = logging.getLogger(__name__)

# payload 中的核心键；其余键视为业务元数据。
CORE_KEYS = frozenset({
    "data",
    "text_lemmatized",
    "hash",
    "created_at",
    "updated_at",
})


def memory_hash(text: str) -> str:
    """正文指纹：用 MD5 便于与既有数据比对。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """当前时间：统一 UTC ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def keyword_text(text: str) -> str:
    """关键词检索文本：小写并折叠空白，供 BM25 或全文索引使用。"""
    return " ".join(str(text).lower().split())


def to_vector_id_int(vector_id: Any) -> Optional[int]:
    """向量库 id 回到业务主键；无法解析时返回 None。"""
    try:
        return int(str(vector_id))
    except (TypeError, ValueError):
        return None

def build_payload(text: str, scope: Dict[str, Any], created_at: Optional[str] = None) -> Dict[str, Any]:
    """组装写入 payload：核心键由本函数负责，作用域由 scope 提供。"""
    stamp = created_at or now_iso()

    query_lemmatized = lemmatize_zh_for_bm25(keyword_text(text))

    payload: Dict[str, Any] = {
        "text": text,
        "text_lemma": query_lemmatized,
        "hash": memory_hash(text),
        "created_at_tz": stamp,
        "updated_at_tz": None,
    }
    payload.update({key: value for key, value in scope.items() if value})
    return payload


def split_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 payload 拆成「核心字段 + 提升字段 + 其余 metadata」。"""
    data = dict(payload or {})
    promoted = {
        key: data.get(key)
        for key in ("user_id", "agent_id", "run_id", "actor_id", "role")
        if data.get(key) is not None
    }
    extra = {key: value for key, value in data.items() if key not in CORE_KEYS and key not in promoted}
    merged = dict(promoted)
    if extra:
        merged["metadata"] = extra
    return merged


class MemoryStore:
    """向量库与关系库的组合封装：统一批量写入、历史审计与失败降级。"""

    def __init__(
            self,
            vector: VectorEngine,
            db: DBEngine,
    ) -> None:
        self.vector = vector
        self.db = db

    def start(self):
        self.vector.start()
        self.db.start()

    @property
    def supports_keyword(self) -> bool:
        """后端是否实现关键词检索：用类方法是否被覆盖来判断。"""
        from loong_memories.storage.memories.base import VectorStoreBase

        store = self.vector
        if store is None:
            return False
        return type(store).keyword_search is not VectorStoreBase.keyword_search

    def _warn(self, message: str, **fields: Any) -> None:
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.warning(f"{message} {detail}" if detail else message)

    def insert_memories(
            self,
            entries: List[Dict[str, Any]],
            session_scope: str,
            messages: Optional[List[Message]] = None,
    ) -> None:
        """批量写记忆：先写向量库再写历史，历史失败只告警。"""
        if not entries:
            return
        vectors = [entry["vector"] for entry in entries]
        ids = [entry["id"] for entry in entries]
        payloads = [entry["payload"] for entry in entries]

        try:
            self.vector.insert(vectors, payloads=payloads, ids=ids)
        except Exception as exc:
            # 批量失败逐条回退，避免单条异常拖垮整批。
            self._warn("记忆批量写入失败，回退逐条写入", error=str(exc))
            failures = 0
            for vector, payload, memory_id in zip(vectors, payloads, ids):
                try:
                    self.vector.insert([vector], payloads=[payload], ids=[memory_id])
                except Exception as single_exc:
                    failures += 1
                    self._warn("记忆写入失败", id=memory_id, error=str(single_exc))
            if failures == len(ids):
                raise StorageError("记忆写入全部失败") from exc

        self._save_history(
            [
                HistoryRecord(
                    id=next_id(),
                    memory_id=entry["id"],
                    old_memory=None,
                    new_memory=entry["payload"].get("text"),
                    event="ADD",
                    created_at=entry["payload"].get("created_at"),
                    updated_at=entry["payload"].get("updated_at"),
                    actor_id=entry["payload"].get("actor_id"),
                    role=entry["payload"].get("role"),
                )
                for entry in entries
            ]
        )

        if messages:
            self._save_messages(messages, session_scope)

    def _save_history(self, records: List[HistoryRecord]) -> None:
        if not records:
            return
        try:
            self.db.batch_add_history(records)
        except Exception as exc:
            self._warn("历史批量写入失败，回退逐条写入", error=str(exc))
            for record in records:
                try:
                    self.db.add_history(record)
                except Exception as single_exc:
                    self._warn("历史写入失败", error=str(single_exc))

    def _save_messages(self, messages: List[Message], session_scope: str) -> None:
        if not messages:
            return
        scoped = [
            message.model_copy(update={"session_scope": session_scope})
            if not message.session_scope
            else message
            for message in messages
        ]
        try:
            self.db.save_messages(scoped)
        except Exception as exc:
            self._warn("消息写入失败", error=str(exc))

    def _build_recall_result(self, explain: bool, scored_results: list[dict[str, Any]]) -> list[Any]:
        # Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "tenant_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"text", "hash", "score", "created_at_tz", "updated_at_tz", "id", "text_lemma",
                                  *promoted_payload_keys}

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}

            if not payload.get("text"):
                continue  # Skip candidates with no payload data

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("text", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at_tz"),
                updated_at=payload.get("updated_at_tz"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if explain and "score_details" in scored:
                memory_item_dict["score_details"] = scored["score_details"]

            original_memories.append(memory_item_dict)

        return list(original_memories)

    def update_memory(self, memory_id: int, vector: List[float], payload: Dict[str, Any],
                      previous: Optional[str]) -> None:
        try:
            self.vector.update(memory_id, vector=vector, payload=payload)
        except Exception as exc:
            raise StorageError("记忆更新失败") from exc
        self._save_history([
            HistoryRecord(
                id=next_id(),
                memory_id=memory_id,
                old_memory=previous,
                new_memory=payload.get("text"),
                event="UPDATE",
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                actor_id=payload.get("actor_id"),
                role=payload.get("role"),
            )
        ])

    def delete_memory(self, memory_id: int, previous: Optional[str], payload: Optional[Dict[str, Any]]) -> None:
        try:
            self.vector.delete(memory_id)
        except Exception as exc:
            raise StorageError("记忆删除失败") from exc
        data = payload or {}
        self._save_history([
            HistoryRecord(
                id=next_id(),
                memory_id=memory_id,
                old_memory=previous,
                new_memory=None,
                event="DELETE",
                created_at=data.get("created_at"),
                updated_at=now_iso(),
                actor_id=data.get("actor_id"),
                role=data.get("role"),
                is_deleted=True,
            )
        ])

    def get_memory(self, memory_id: int) -> Optional[VectorStoreResult]:
        try:
            return self.vector.get(memory_id)
        except Exception as exc:
            # 后端故障与「不存在」必须区分：前者上抛，后者返回 None。
            raise StorageError("记忆读取失败") from exc

    def load_history(self, session_scope: str, limit: int) -> List[Dict[str, Any]]:
        """取会话最近消息，转成提示词可直接消费的字典列表。"""
        if limit <= 0:
            return []
        try:
            messages = self.db.get_last_messages(session_scope, limit=limit)
        except Exception as exc:
            self._warn("历史读取失败", error=str(exc))
            return []
        return [{"role": message.role, "content": message.content} for message in messages]

    def recall(self, query: str, vector: List[float], filters: Dict[str, Any], top_k: int, threshold: float = 0.1, explain: bool = False) -> List[Dict[str,Any]]:
        """双路召回, 打分："""
        internal_limit = max(top_k * 4, 60)
        try:
            semantic_results = list(self.vector.search(query, vector, top_k=internal_limit, filters=filters) or [])
        except Exception as exc:
            raise StorageError("语义检索失败") from exc

        bm25_scores = {}
        if self.supports_keyword:
            keyword_query = keyword_text(query)
            query_lemmatized = lemmatize_zh_for_bm25(keyword_query)

            try:
                keyword_results = self.vector.keyword_search(query=query_lemmatized, top_k=internal_limit, filters=filters) or []
            except Exception as exc:
                self._warn("关键词检索失败，仅用语义结果", error=str(exc))
                keyword_results = []

            if len(keyword_results) > 0:
                midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
                for mem in keyword_results:
                    mem_id = mem.id
                    raw_score = mem.score
                    if raw_score and raw_score > 0:
                        bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # TODO : Compute entity boosts
        entity_boosts = {}

        candidates = []
        for mem in semantic_results:
            payload = {**mem.model_dump(exclude_none=True,exclude={"metadata"}), **mem.metadata}

            mem_id = mem.id
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": payload,
            })

        # Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=top_k,
            explain=explain,
        )

        return self._build_recall_result(explain, scored_results)


class AsyncMemoryStore(MemoryStore):
    """向量库与关系库的组合封装：统一批量写入、历史审计与失败降级。"""

    def __init__(self, vector: AsyncVectorEngine, db: AsyncDBEngine) -> None:
        super().__init__(vector, db)
        self.vector = vector
        self.db = db

    async def start(self):
        await self.vector.start()
        await self.db.start()

    async def insert_memories(
            self,
            entries: List[Dict[str, Any]],
            session_scope: str,
            messages: Optional[List[Message]] = None,
    ) -> None:
        """批量写记忆：先写向量库再写历史，历史失败只告警。"""
        if not entries:
            return
        vectors = [entry["vector"] for entry in entries]
        ids = [entry["id"] for entry in entries]
        payloads = [entry["payload"] for entry in entries]

        try:
            await self.vector.insert(vectors, payloads=payloads, ids=ids)
        except Exception as exc:
            # 批量失败逐条回退，避免单条异常拖垮整批。
            self._warn("记忆批量写入失败，回退逐条写入", error=str(exc))
            failures = 0
            for vector, payload, memory_id in zip(vectors, payloads, ids):
                try:
                    await self.vector.insert([vector], payloads=[payload], ids=[memory_id])
                except Exception as single_exc:
                    failures += 1
                    self._warn("记忆写入失败", id=memory_id, error=str(single_exc))
            if failures == len(ids):
                raise StorageError("记忆写入全部失败") from exc

        await self._save_history(
            [
                HistoryRecord(
                    id=next_id(),
                    memory_id=entry["id"],
                    old_memory=None,
                    new_memory=entry["payload"].get("data"),
                    event="ADD",
                    created_at=entry["payload"].get("created_at"),
                    updated_at=entry["payload"].get("updated_at"),
                    actor_id=entry["payload"].get("actor_id"),
                    role=entry["payload"].get("role"),
                )
                for entry in entries
            ]
        )

        if messages:
            await self._save_messages(messages, session_scope)

    async def _save_history(self, records: List[HistoryRecord]) -> None:
        if not records:
            return
        try:
            await self.db.batch_add_history(records)
        except Exception as exc:
            self._warn("历史批量写入失败，回退逐条写入", error=str(exc))
            for record in records:
                try:
                    await self.db.add_history(record)
                except Exception as single_exc:
                    self._warn("历史写入失败", error=str(single_exc))

    async def _save_messages(self, messages: List[Message], session_scope: str) -> None:
        if not messages:
            return
        scoped = [
            message.model_copy(update={"session_scope": session_scope})
            if not message.session_scope
            else message
            for message in messages
        ]
        try:
            await self.db.save_messages(scoped)
        except Exception as exc:
            self._warn("消息写入失败", error=str(exc))

    async def update_memory(self, memory_id: int, vector: List[float], payload: Dict[str, Any],
                            previous: Optional[str]) -> None:
        try:
            await self.vector.update(memory_id, vector=vector, payload=payload)
        except Exception as exc:
            raise StorageError("记忆更新失败") from exc
        await self._save_history([
            HistoryRecord(
                id=next_id(),
                memory_id=memory_id,
                old_memory=previous,
                new_memory=payload.get("text"),
                event="UPDATE",
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                actor_id=payload.get("actor_id"),
                role=payload.get("role"),
            )
        ])

    async def delete_memory(self, memory_id: int, previous: Optional[str], payload: Optional[Dict[str, Any]]) -> None:
        try:
            await self.vector.delete(memory_id)
        except Exception as exc:
            raise StorageError("记忆删除失败") from exc
        data = payload or {}
        await self._save_history([
            HistoryRecord(
                id=next_id(),
                memory_id=memory_id,
                old_memory=previous,
                new_memory=None,
                event="DELETE",
                created_at=data.get("created_at"),
                updated_at=now_iso(),
                actor_id=data.get("actor_id"),
                role=data.get("role"),
                is_deleted=True,
            )
        ])

    async def get_memory(self, memory_id: int) -> Optional[VectorStoreResult]:
        try:
            return await self.vector.get(memory_id)
        except Exception as exc:
            # 后端故障与「不存在」必须区分：前者上抛，后者返回 None。
            raise StorageError("记忆读取失败") from exc

    async def load_history(self, session_scope: str, limit: int) -> List[Dict[str, Any]]:
        """取会话最近消息，转成提示词可直接消费的字典列表。"""
        if limit <= 0:
            return []
        try:
            messages = await self.db.get_last_messages(session_scope, limit=limit)
        except Exception as exc:
            self._warn("历史读取失败", error=str(exc))
            return []
        return [{"role": message.role, "content": message.content} for message in messages]

    async def recall(self, query: str, vector: List[float], filters: Dict[str, Any], top_k: int, threshold: float = 0.1, explain: bool = False) -> List[Dict[str,Any]]:
        """双路召回, 打分："""
        internal_limit = max(top_k * 4, 60)
        try:
            semantic_results = await self.vector.search(query, vector, top_k=internal_limit, filters=filters) or []
        except Exception as exc:
            raise StorageError("语义检索失败") from exc

        bm25_scores = {}
        if self.supports_keyword:
            keyword_query = keyword_text(query)
            query_lemmatized = lemmatize_zh_for_bm25(keyword_query)

            try:
                keyword_results = await self.vector.keyword_search(query=query_lemmatized, top_k=internal_limit,filters=filters) or []
            except Exception as exc:
                self._warn("关键词检索失败，仅用语义结果", error=str(exc))
                keyword_results = []

            if len(keyword_results) > 0:
                midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
                for mem in keyword_results:
                    mem_id = mem.id
                    raw_score = mem.score
                    if raw_score and raw_score > 0:
                        bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)


        # TODO : Compute entity boosts
        entity_boosts = {}

        candidates = []
        for mem in semantic_results:
            payload = {**mem.model_dump(exclude_none=True, exclude={"metadata"}), **mem.metadata}

            mem_id = mem.id
            candidates.append({
                "id": mem_id,
                "score": mem.score,
                "payload": payload,
            })

        # Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=top_k,
            explain=explain,
        )

        return self._build_recall_result(explain, scored_results)


