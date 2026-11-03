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

from loong_memories.core.config_nlp import NlpSettings
from loong_memories.core.exceptions import StorageError
from loong_memories.memory.models import MemoryItem
from loong_memories.nlp.lemmatization import lemmatize_zh_for_bm25
from loong_memories.nlp.spacy import load_nlp_lemma
from loong_memories.storage.memories.base import VectorStoreResult
from loong_memories.storage.memories.vector_engine import VectorEngine, AsyncVectorEngine
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine
from loong_memories.storage.message.schema import HistoryRecord, Message
from loong_memories.util.datetime_util import now_iso, now_datetime
from loong_memories.util.numberic_util import str_to_int
from loong_memories.util.scoring import get_bm25_params, normalize_bm25, score_and_rank
from loong_memories.util.snowflake import next_id
from loong_memories.util.str_util import memory_hash

logger = logging.getLogger(__name__)

# payload 中的核心键；其余键视为业务元数据。
CORE_KEYS = frozenset({
    "id",
    "tenant_id",
    "user_id",
    "text",
    "text_lemmatized",
    "hash",
    "created_at_tz",
    "updated_at_tz",
})



def keyword_text(text: str) -> str:
    """关键词检索文本：小写并折叠空白，供 BM25 或全文索引使用。"""
    return " ".join(str(text).lower().split())


def to_vector_id_int(vector_id: Any) -> Optional[int]:
    """向量库 id 回到业务主键；无法解析时返回 None。"""
    try:
        return int(str(vector_id))
    except (TypeError, ValueError):
        return None


def split_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 payload 拆成「核心字段 + 提升字段 + 其余 metadata」。"""
    data = dict(payload or {})
    promoted = {
        key: data.get(key)
        for key in ("agent_id", "run_id", "actor_id", "role")
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
            vector_engine: VectorEngine,
            db_engine: DBEngine,
            nlp_setting: Optional[NlpSettings] = None
    ) -> None:
        self.vector_engine = vector_engine
        self.db_engine = db_engine
        self.nlp_setting = nlp_setting

    def start(self):
        self.vector_engine.start()
        self.db_engine.start()
        self._loading_nlp()

    def _loading_nlp(self):
        if self.nlp_setting and self.nlp_setting.enable:
            load_nlp_lemma(self.nlp_setting.nlp_name)

    @property
    def supports_keyword(self) -> bool:
        """后端是否实现关键词检索：用类方法是否被覆盖来判断。"""
        from loong_memories.storage.memories.base import VectorStoreBase

        store = self.vector_engine
        if self.nlp_setting is None or self.nlp_setting.enable is False:
            return False

        return type(store).keyword_search is not VectorStoreBase.keyword_search

    def build_payload(self, text: str, new_metadata: Dict[str, Any], created_at: Optional[str] = None) -> Dict[str, Any]:
        """组装写入 payload：核心键由本函数负责，作用域由 scope 提供。"""
        stamp = created_at or now_iso()

        query_lemmatized = None
        if self.nlp_setting is not None and self.nlp_setting.enable:
            query_lemmatized = lemmatize_zh_for_bm25(keyword_text(text), self.nlp_setting.nlp_name)

        payload: Dict[str, Any] = {
            "text": text,
            "text_lemma": query_lemmatized,
            "hash": memory_hash(text),
            "created_at_tz": stamp,
            "updated_at_tz": None,
        }
        payload.update({key: value for key, value in new_metadata.items() if value})
        return payload

    def _warn(self, message: str, **fields: Any) -> None:
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.warning(f"{message} {detail}" if detail else message)

    def _save_history(self, records: List[HistoryRecord]) -> None:
        if not records:
            return
        try:
            self.db_engine.batch_add_history(records)
        except Exception as exc:
            self._warn("历史写入失败", error=str(exc))

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
            self.db_engine.save_messages(scoped)
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
                tenant_id=scored["tenant_id"],
                user_id=scored["user_id"],
                memory=payload.get("text", ""),
                hash=payload.get("hash"),
                created_at_tz=payload.get("created_at_tz"),
                updated_at_tz=payload.get("updated_at_tz"),
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
            self.vector_engine.insert(vectors, payloads=payloads, ids=ids)
        except Exception as exc:
            raise StorageError("记忆写入全部失败") from exc

        records: List[HistoryRecord] = []
        for entry in entries:

            tenant_id = str_to_int(entry.get("tenant_id"))
            user_id = str_to_int(entry.get("user_id"))

            record = HistoryRecord(
                    id=next_id(),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_id=entry["id"],
                    old_memory=None,
                    new_memory=entry["payload"].get("text"),
                    event="ADD",
                    created_at_tz=entry["payload"].get("created_at_tz"),
                    updated_at_tz=entry["payload"].get("updated_at_tz"),
                    actor_id=entry["payload"].get("actor_id"),
                    role=entry["payload"].get("role"),
                )

            records.append(record)

        self._save_history(records)

        if messages:
            self._save_messages(messages, session_scope)

    def update_memory(self, memory_id: int, tenant_id: int, vector: List[float], payload: Dict[str, Any],
                      previous: Optional[str], user_id: Optional[int] = None) -> None:
        '''按 id 更新记忆：租户与用户作用域透传给向量引擎。'''
        try:
            self.vector_engine.update(memory_id, tenant_id, user_id, vector=vector, payload=payload)
        except Exception as exc:
            raise StorageError("记忆更新失败") from exc
        self._save_history([
            HistoryRecord(
                id=next_id(),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=memory_id,
                old_memory=previous,
                new_memory=payload.get("text"),
                event="UPDATE",
                created_at_tz=payload.get("created_at_tz"),
                updated_at_tz=payload.get("updated_at_tz"),
                actor_id=payload.get("actor_id"),
                role=payload.get("role"),
            )
        ])

    def delete_memory(self, memory_id: int, tenant_id: int, previous: Optional[str],
                      payload: Optional[Dict[str, Any]], user_id: Optional[int] = None) -> None:
        '''按 id 删除记忆；向量引擎当前只接受 id，租户维度待引擎层放开后再透传。'''
        try:
            self.vector_engine.delete(memory_id,tenant_id, user_id)
        except Exception as exc:
            raise StorageError("记忆删除失败") from exc
        data = payload or {}
        self._save_history([
            HistoryRecord(
                id=next_id(),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=memory_id,
                old_memory=previous,
                new_memory=None,
                event="DELETE",
                created_at_tz=data.get("created_at_tz"),
                updated_at_tz=now_datetime(),
                actor_id=data.get("actor_id"),
                role=data.get("role"),
                is_deleted=True,
            )
        ])

    def get_memory(self, memory_id: int, tenant_id: int, user_id: Optional[int] = None) -> Optional[VectorStoreResult]:
        '''按 id 读取记忆：租户必填、用户可选，二者一并作为过滤条件。'''
        try:
            return self.vector_engine.get(memory_id, tenant_id, user_id=user_id)
        except Exception as exc:
            # 后端故障与「不存在」必须区分：前者上抛，后者返回 None。
            raise StorageError("记忆读取失败") from exc

    def load_history(self, session_scope: str, tenant_id: int, limit: int,
                     user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """取会话最近消息，转成提示词可直接消费的字典列表。

        消息表暂未按租户分列，tenant_id 与 user_id 先统一签名，待表结构扩展后再下发。
        """
        if limit <= 0:
            return []
        try:
            messages = self.db_engine.get_last_messages(session_scope, tenant_id, limit, user_id)
        except Exception as exc:
            self._warn("历史读取失败", error=str(exc))
            return []
        return [{"role": message.role, "content": message.content} for message in messages]

    def recall(self,
               query: str,
               tenant_id: int,
               user_id: Optional[int] = None,
               *,
               vector: List[float],
               filter: Optional[Dict[str, Any]] = None,
               top_k: int = 5,
               threshold: float = 0.1,
               explain: bool = False) -> List[Dict[str,Any]]:
        """双路召回, 打分："""

        internal_limit = max(top_k * 4, 60)
        try:
            semantic_results = list(self.vector_engine.search(query, vector, tenant_id, user_id, internal_limit, filter) or [])
        except Exception as exc:
            raise StorageError("语义检索失败") from exc

        bm25_scores = {}
        if self.supports_keyword:
            keyword_query = keyword_text(query)
            query_lemmatized = lemmatize_zh_for_bm25(keyword_query,self.nlp_setting.nlp_name)

            try:
                keyword_results = self.vector_engine.keyword_search(query=query_lemmatized, tenant_id=tenant_id, user_id=user_id, top_k=internal_limit, filter=filter) or []
            except Exception as exc:
                self._warn("关键词检索失败，仅用语义结果", error=str(exc))
                keyword_results = []

            if len(keyword_results) > 0:
                midpoint, steepness = get_bm25_params(keyword_query, lemmatized=query_lemmatized)
                for mem in keyword_results:
                    mem_id = mem.id
                    raw_score = mem.score
                    if raw_score and raw_score > 0:
                        bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)

        # TODO : Compute entity boosts
        entity_boosts = {}

        candidates = []
        for mem in semantic_results:
            payload = {**mem.model_dump(exclude_none=True,exclude={"metadata"}), **(mem.metadata or {})}

            mem_id = mem.id
            candidates.append({
                "id": mem_id,
                "tenant_id": mem.tenant_id,
                "user_id": mem.user_id,
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

    def __init__(self, vector_engine: AsyncVectorEngine, db_engine: AsyncDBEngine,
                 nlp_setting: Optional[NlpSettings] = None) -> None:
        super().__init__(vector_engine, db_engine, nlp_setting)
        self.vector_engine = vector_engine
        self.db_engine = db_engine

    async def start(self):
        await self.vector_engine.start()
        await self.db_engine.start()
        self._loading_nlp()

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
            await self.vector_engine.insert(vectors, payloads=payloads, ids=ids)
        except Exception as exc:
            raise StorageError("记忆写入全部失败") from exc

        records: List[HistoryRecord] = []
        for entry in entries:
            tenant_id = str_to_int(entry.get("tenant_id"))
            user_id = str_to_int(entry.get("user_id"))

            record = HistoryRecord(
                id=next_id(),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=entry["id"],
                old_memory=None,
                new_memory=entry["payload"].get("text"),
                event="ADD",
                created_at_tz=entry["payload"].get("created_at_tz"),
                updated_at_tz=entry["payload"].get("updated_at_tz"),
                actor_id=entry["payload"].get("actor_id"),
                role=entry["payload"].get("role"),
            )

            records.append(record)

        await self._save_history(records)

        if messages:
            await self._save_messages(messages, session_scope)

    async def _save_history(self, records: List[HistoryRecord]) -> None:
        if not records:
            return
        try:
            await self.db_engine.batch_add_history(records)
        except Exception as exc:
            self._warn("历史写入失败", error=str(exc))


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
            await self.db_engine.save_messages(scoped)
        except Exception as exc:
            self._warn("消息写入失败", error=str(exc))

    async def update_memory(self, memory_id: int, tenant_id: int, vector: List[float], payload: Dict[str, Any],
                            previous: Optional[str], user_id: Optional[int] = None) -> None:
        '''按 id 更新记忆：租户与用户作用域透传给向量引擎。'''
        try:
            await self.vector_engine.update(memory_id, tenant_id, user_id, vector=vector, payload=payload)
        except Exception as exc:
            raise StorageError("记忆更新失败") from exc
        await self._save_history([
            HistoryRecord(
                id=next_id(),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=memory_id,
                old_memory=previous,
                new_memory=payload.get("text"),
                event="UPDATE",
                created_at_tz=payload.get("created_at_tz"),
                updated_at_tz=payload.get("updated_at_tz"),
                actor_id=payload.get("actor_id"),
                role=payload.get("role"),
            )
        ])

    async def delete_memory(self, memory_id: int, tenant_id: int, previous: Optional[str],
                            payload: Optional[Dict[str, Any]], user_id: Optional[int] = None) -> None:
        '''按 id 删除记忆；向量引擎当前只接受 id，租户维度待引擎层放开后再透传。'''
        try:
            await self.vector_engine.delete(memory_id,tenant_id, user_id)
        except Exception as exc:
            raise StorageError("记忆删除失败") from exc
        data = payload or {}
        await self._save_history([
            HistoryRecord(
                id=next_id(),
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=memory_id,
                old_memory=previous,
                new_memory=None,
                event="DELETE",
                created_at_tz=data.get("created_at_tz"),
                updated_at_tz=now_datetime(),
                actor_id=data.get("actor_id"),
                role=data.get("role"),
                is_deleted=True,
            )
        ])

    async def get_memory(self, memory_id: int, tenant_id: int, user_id: Optional[int] = None) -> Optional[VectorStoreResult]:
        '''按 id 读取记忆：租户必填、用户可选，二者一并作为过滤条件。'''
        try:
            return await self.vector_engine.get(memory_id, tenant_id, user_id=user_id)
        except Exception as exc:
            # 后端故障与「不存在」必须区分：前者上抛，后者返回 None。
            raise StorageError("记忆读取失败") from exc

    async def load_history(self, session_scope: str, tenant_id: int, limit: int,
                           user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """取会话最近消息，转成提示词可直接消费的字典列表。

        消息表暂未按租户分列，tenant_id 与 user_id 先统一签名，待表结构扩展后再下发。
        """
        if limit <= 0:
            return []
        try:
            messages = await self.db_engine.get_last_messages(session_scope, tenant_id, limit, user_id)
        except Exception as exc:
            self._warn("历史读取失败", error=str(exc))
            return []
        return [{"role": message.role, "content": message.content} for message in messages]

    async def recall(self,
            query: str,
            tenant_id: int,
            user_id: Optional[int] = None,
            *,
            vector: List[float],
            filter: Optional[Dict[str, Any]] = None,
            top_k: int = 5,
            threshold: float = 0.1,
            explain: bool = False
            ) -> List[Dict[str,Any]]:
        """双路召回, 打分："""

        internal_limit = max(top_k * 4, 60)
        try:
            semantic_results = await self.vector_engine.search(query, vector,tenant_id, user_id, internal_limit, filter) or []
        except Exception as exc:
            raise StorageError("语义检索失败") from exc

        bm25_scores = {}
        if self.supports_keyword:
            keyword_query = keyword_text(query)
            query_lemmatized = lemmatize_zh_for_bm25(keyword_query,self.nlp_setting.nlp_name)

            try:
                keyword_results = await self.vector_engine.keyword_search(query=query_lemmatized, tenant_id=tenant_id, user_id=user_id, top_k=internal_limit, filter=filter) or []
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
            payload = {**mem.model_dump(exclude_none=True, exclude={"metadata"}), **(mem.metadata or {})}

            mem_id = mem.id
            candidates.append({
                "id": mem_id,
                "tenant_id": mem.tenant_id,
                "user_id": mem.user_id,
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


