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
 记忆层处理器：装配四个下层组件，实现 add、search、update、delete 四条同步流程。

 分工：本文件只做流程编排与结果组装；存储细节在 stores.py，
 作用域校验在 scope.py，提示词在 prompts.py。
"""


from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from loong_memories.core.config_memory import MemorySettings
from loong_memories.core.exceptions import ConfigError, DataError, EmbeddingError, LLMError
from loong_memories.memory.base import MemoryBase
from loong_memories.memory.models import AddResult, DeleteResult, MemoryItem, SearchResult, UpdateResult
from loong_memories.memory.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    generate_additive_extraction_prompt,
)
from loong_memories.memory.scope import (
    build_scope,
    session_scope,
    strip_identity_keys,
    validate_query,
    validate_search_params,
)
from loong_memories.memory.stores import (
    MemoryStore,
    build_payload,
    keyword_text,
    memory_hash,
    now_iso,
    split_payload,
    to_vector_id_int, AsyncMemoryStore,
)

from loong_memories.embedding.embedder_handler import EmbedderHandler
from loong_memories.llm.llm_handler import LLMHandler
from loong_memories.storage.message.base import BaseDB
from loong_memories.storage.memories.vector_engine import VectorEngine, AsyncVectorEngine
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine
from loong_memories.storage.message.schema import Message
from loong_memories.util.snowflake import next_id

logger = logging.getLogger(__name__)


# 检索与写入的默认参数：memory 分区不再逐项暴露，集中在此处统一调整。
DEFAULT_THRESHOLD = 0.1
DEFAULT_TOP_K = 20
DEFAULT_CONTEXT_TOP_K = 10
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_HISTORY_MESSAGES = 10


class MemoryHandler(MemoryBase):
    """记忆层处理器：四条流程均为同步阻塞，不引入并发。"""

    def __init__(
        self,
        settings: MemorySettings,
        embedder: EmbedderHandler,
        llm: LLMHandler,
        db: DBEngine,
        vector: VectorEngine,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.db = db
        self.vector = vector
        self._validate()
        self.store = MemoryStore(vector, db=db)


    # ---------------- 内部辅助 ----------------

    def _validate(self) -> None:
        """构造期校验：向量库、嵌入器、大模型与关系库都是硬依赖。"""
        if self.vector is None:
            raise ConfigError("记忆层需要向量库：请启用 vector 分区")
        if self.embedder is None:
            raise ConfigError("记忆层需要嵌入器：请启用 embedding 分区或注入 embedder")
        if self.llm is None:
            raise ConfigError("记忆层需要大模型进行事实抽取：请启用 llm 分区")
        if self.db is None:
            raise ConfigError("记忆层需要关系库记录历史与消息：请启用 db 分区")

    def _log(self, message: str, **fields: Any) -> None:
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info(f"{message} {detail}" if detail else message)


    @staticmethod
    def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
        """把字符串、单条字典或字典列表归一为统一的消息列表。"""
        if isinstance(messages, str):
            items: List[Any] = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            items = [messages]
        elif isinstance(messages, list):
            items = messages
        else:
            raise DataError("messages 必须是字符串、字典或字典列表")

        normalized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise DataError("消息必须是字典")
            role = item.get("role")
            content = item.get("content")
            if content is None:
                raise DataError("消息缺少 content 字段")
            entry: Dict[str, Any] = {"role": role or "user", "content": str(content)}
            if item.get("name"):
                entry["name"] = item["name"]
            normalized.append(entry)
        if not normalized:
            raise DataError("messages 不能为空")
        return normalized


    def _embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入：失败时抛 EmbeddingError，不静默降级。"""
        if not texts:
            return []
        try:
            return self.embedder.encode(texts)
        except Exception as exc:
            raise EmbeddingError("记忆向量化失败") from exc


    def _extract_facts(
        self,
        messages: List[Dict[str, Any]],
        existing: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        instructions: str = "",
        agent_scoped: bool = False,
    ) -> List[str]:
        """调用大模型抽取事实；提示词与 mem0 一致，模型失败抛 LLMError。"""
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX
        prompt = generate_additive_extraction_prompt(
            existing_memories=existing,
            new_messages=messages,
            last_k_messages=history,
            custom_instructions=instructions,
        )
        try:
            response = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ])
        except Exception as exc:
            raise LLMError("记忆抽取失败") from exc
        return self._parse_extraction(response)


    def _parse_extraction(self, response: Any) -> List[str]:
        """解析模型响应：只取 memory 数组中的 text，非 JSON 视为未抽取。"""
        if not response or not str(response).strip():
            return []
        text = str(response).strip()
        # 兼容模型偶尔包裹代码块的情况：截取首个花括号到末个花括号。
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start:end + 1]
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            self._warn("抽取响应不是合法 JSON，按未抽取处理")
            return []
        items = data.get("memory") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        facts: List[str] = []
        for item in items:
            candidate = item.get("text") if isinstance(item, dict) else item
            if isinstance(candidate, str) and candidate.strip():
                facts.append(candidate.strip())
        return facts


    def _warn(self, message: str, **fields: Any) -> None:
        detail = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.warning(f"{message} {detail}" if detail else message)


    # ---------------- 流程一：写入 ----------------

    def add(
        self,
        messages: Any,
        *,
        user_id: Optional[Any] = None,
        agent_id: Optional[Any] = None,
        run_id: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AddResult:
        """写入记忆：先由大模型抽取事实，去重后批量落库。"""
        scope, query = build_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )
        normalized = self._normalize_messages(messages)

        # 先召回既有记忆：既作为抽取上下文，也用于库内去重。
        existing, existing_hashes = self._recall_existing(normalized, query)
        session = session_scope(query)
        history = self.store.load_history(session, DEFAULT_HISTORY_MESSAGES)
        # 与 mem0 一致：仅 agent 作用域（有 agent_id 且无 user_id）附加实体上下文后缀。
        agent_scoped = bool(query.get("agent_id")) and not query.get("user_id")
        facts = self._extract_facts(normalized, existing, history, agent_scoped=agent_scoped)

        recorded = self._persist_facts(facts, scope, existing_hashes, session, normalized)
        self._log("记忆写入完成", added=len(recorded))
        return AddResult(results=recorded)


    def _recall_existing(self, messages: List[Dict[str, Any]], query: Dict[str, Any]):
        """召回既有记忆：返回 [id, text] 列表与指纹集合，供提示词与去重共用。"""
        keyword = " ".join(message["content"] for message in messages if message.get("content"))
        vector = self._embed([keyword])[0]
        existing = self.store.recall(keyword, vector, query, DEFAULT_CONTEXT_TOP_K)
        items: List[Dict[str, Any]] = []
        hashes = set()
        for item in existing:
            payload = item.payload or {}
            text = payload.get("data")
            if not isinstance(text, str) or not text:
                continue
            # 提示词要求既有记忆为 [{"id": ..., "text": ...}] 结构。
            items.append({"id": str(item.id), "text": text})
            digest = payload.get("hash")
            if isinstance(digest, str) and digest:
                hashes.add(digest)
        return items, hashes


    def _persist_facts(
        self,
        facts: List[str],
        scope: Dict[str, Any],
        existing_hashes: set,
        session: str,
        messages: List[Dict[str, Any]],
    ) -> List[MemoryItem]:
        """去重、批量嵌入并落库；返回本次真正新增的条目。"""
        unique: List[str] = []
        seen = set(existing_hashes)
        for text in facts:
            digest = memory_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(text)

        if not unique:
            # 没有新增也要保存消息，保证会话上下文连续。
            self.store._save_messages(self._to_message_objects(messages, session), session)
            return []

        vectors = self._embed(unique)
        entries = []
        results: List[MemoryItem] = []
        for text, vector in zip(unique, vectors):
            memory_id = next_id()
            payload = build_payload(text, scope)
            entries.append({"id": memory_id, "vector": vector, "payload": payload})
            results.append(self._to_item(memory_id, payload, None))


        self.store.insert_memories(entries, session, self._to_message_objects(messages, session))
        return results


    @staticmethod
    def _to_message_objects(messages: List[Dict[str, Any]], session: str) -> List[Message]:
        """消息字典转存储对象；会话键由作用域统一填充。"""
        return [
            Message(
                session_scope=session,
                role=message.get("role"),
                content=message.get("content"),
                name=message.get("name"),
                created_at=now_iso(),
            )
            for message in messages
        ]


    @staticmethod
    def _to_item(memory_id: int, payload: Optional[Dict[str, Any]], score: Optional[float]) -> MemoryItem:
        """payload 转对外条目：核心字段直取，其余归入 metadata。"""
        data = payload or {}
        extra = split_payload(data)
        return MemoryItem(
            id=memory_id,
            memory=str(data.get("data") or ""),
            score=score,
            hash=data.get("hash"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            **extra,
        )


    # ---------------- 流程二：检索 ----------------

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SearchResult:
        """检索记忆：语义与关键词双路召回取并集，融合打分后过滤与截断。"""
        limit = DEFAULT_TOP_K if top_k is None else top_k
        floor = DEFAULT_THRESHOLD if threshold is None else threshold
        validate_search_params(floor, limit)
        text = validate_query(query)
        scope = self._build_filters(filters)

        recall_k = max(limit * 4, 60)
        vector = self._embed([text])[0]
        candidates = self.store.recall(text, vector, scope, recall_k)
        results = self._fuse_and_rank(candidates, floor, limit)
        self._log("记忆检索完成", candidates=len(candidates), hits=len(results))
        return SearchResult(results=results)


    @staticmethod
    def _build_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """查询侧作用域：只保留三个实体键，其余高级过滤条件原样透传。"""
        source = dict(filters or {})
        scope: Dict[str, Any] = {}
        for name in ("user_id", "agent_id", "run_id"):
            value = source.pop(name, None)
            if value is not None:
                scope[name] = str(value).strip()
        if not any(scope.get(name) for name in ("user_id", "agent_id", "run_id")):
            raise DataError("filters 需要 user_id、agent_id、run_id 至少一个")
        scope.update(source)
        return scope


    def _fuse_and_rank(self, candidates, threshold: float, limit: int) -> List[MemoryItem]:
        """融合打分：语义分与关键词分加权求和，先按阈值过滤再截断。"""
        weights = DEFAULT_KEYWORD_WEIGHT if self.store.supports_keyword else 0.0
        scored = []
        for item in candidates:
            payload = item.payload or {}
            if not payload.get("data"):
                continue
            semantic = float(item.score or 0.0)
            # 关键词分数已由后端归一，取不到时按零处理。
            keyword = float(payload.get("score") or 0.0)
            score = semantic + weights * keyword
            scored.append((score, item, payload))


        # 先过滤阈值，再截断，避免高分的被低分的挤掉。
        kept = [entry for entry in scored if entry[0] >= threshold]
        kept.sort(key=lambda entry: entry[0], reverse=True)

        results: List[MemoryItem] = []
        for score, item, payload in kept[:limit]:
            memory_id = to_vector_id_int(item.id)
            if memory_id is None:
                continue
            results.append(self._to_item(memory_id, payload, score))
        return results


    # ---------------- 流程三：更新 ----------------

    def update(
        self,
        memory_id: int,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UpdateResult:
        """更新记忆：正文变化才重算向量，元数据合并后覆盖旧值。"""
        if text is None and metadata is None:
            raise DataError("text 与 metadata 至少提供一个")
        current = self.store.get_memory(memory_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload, previous, changed = self._merge_payload(current.payload, text, metadata)

        if changed:
            vector = self._embed([payload["data"]])[0]
        else:
            # 正文未变：传 None 让后端保留原向量，避免无谓的嵌入调用。
            vector = None

        self.store.update_memory(memory_id, vector, payload, previous)
        self._log("记忆更新完成", id=memory_id, text_changed=changed)
        return UpdateResult(result=self._to_item(memory_id, payload, None))


    def _merge_payload(
        self,
        payload: Optional[Dict[str, Any]],
        text: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ):
        """在旧 payload 上合并新值：保留 created_at，刷新 updated_at 与指纹。"""
        merged = dict(payload or {})
        previous = merged.get("data")
        if metadata:
            merged.update(strip_identity_keys(metadata, merged, context="update"))

        new_text = text if text is not None else previous
        if new_text is None:
            raise DataError("记忆没有正文，无法更新")
        changed = new_text != previous
        if changed:
            merged["data"] = new_text
            merged["hash"] = memory_hash(new_text)
            merged["text_lemmatized"] = keyword_text(new_text)
        merged.setdefault("created_at", now_iso())
        merged["updated_at"] = now_iso()
        return merged, previous, changed


    # ---------------- 流程四：删除 ----------------

    def delete(self, memory_id: int) -> DeleteResult:
        """删除记忆：向量库物理删除，历史表写入 DELETE 审计记录。"""
        current = self.store.get_memory(memory_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload = current.payload or {}
        self.store.delete_memory(memory_id, payload.get("data"), payload)
        self._log("记忆删除完成", id=memory_id)
        return DeleteResult(id=memory_id)



class AsyncMemoryHandler(MemoryHandler):
    """记忆层处理器：四条流程均为同步阻塞，不引入并发。"""

    def __init__(self, settings: MemorySettings, embedder: EmbedderHandler, llm: LLMHandler, db: AsyncDBEngine,
                 vector: AsyncVectorEngine) -> None:
        super().__init__(settings, embedder, llm, db, vector)
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.db = db
        self.vector = vector
        self._validate()
        self.store = AsyncMemoryStore(vector, db=db)


    async def add(
        self,
        messages: Any,
        *,
        user_id: Optional[Any] = None,
        agent_id: Optional[Any] = None,
        run_id: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AddResult:
        """写入记忆：先由大模型抽取事实，去重后批量落库。"""
        scope, query = build_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )
        normalized = self._normalize_messages(messages)

        # 先召回既有记忆：既作为抽取上下文，也用于库内去重。
        existing, existing_hashes = await self._recall_existing(normalized, query)
        session = session_scope(query)
        history = await self.store.load_history(session, DEFAULT_HISTORY_MESSAGES)
        # 与 mem0 一致：仅 agent 作用域（有 agent_id 且无 user_id）附加实体上下文后缀。
        agent_scoped = bool(query.get("agent_id")) and not query.get("user_id")
        facts = self._extract_facts(normalized, existing, history, agent_scoped=agent_scoped)

        recorded = self._persist_facts(facts, scope, existing_hashes, session, normalized)
        self._log("记忆写入完成", added=len(recorded))
        return AddResult(results=recorded)


    async def _recall_existing(self, messages: List[Dict[str, Any]], query: Dict[str, Any]):
        """召回既有记忆：返回 [id, text] 列表与指纹集合，供提示词与去重共用。"""
        keyword = " ".join(message["content"] for message in messages if message.get("content"))
        vector = self._embed([keyword])[0]
        existing = await self.store.recall(keyword, vector, query, DEFAULT_CONTEXT_TOP_K)
        items: List[Dict[str, Any]] = []
        hashes = set()
        for item in existing:
            payload = item.payload or {}
            text = payload.get("data")
            if not isinstance(text, str) or not text:
                continue
            # 提示词要求既有记忆为 [{"id": ..., "text": ...}] 结构。
            items.append({"id": str(item.id), "text": text})
            digest = payload.get("hash")
            if isinstance(digest, str) and digest:
                hashes.add(digest)
        return items, hashes


    async def _persist_facts(
        self,
        facts: List[str],
        scope: Dict[str, Any],
        existing_hashes: set,
        session: str,
        messages: List[Dict[str, Any]],
    ) -> List[MemoryItem]:
        """去重、批量嵌入并落库；返回本次真正新增的条目。"""
        unique: List[str] = []
        seen = set(existing_hashes)
        for text in facts:
            digest = memory_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(text)

        if not unique:
            # 没有新增也要保存消息，保证会话上下文连续。
            await self.store._save_messages(self._to_message_objects(messages, session), session)
            return []

        vectors = self._embed(unique)
        entries = []
        results: List[MemoryItem] = []
        for text, vector in zip(unique, vectors):
            memory_id = next_id()
            payload = build_payload(text, scope)
            entries.append({"id": memory_id, "vector": vector, "payload": payload})
            results.append(self._to_item(memory_id, payload, None))


        await self.store.insert_memories(entries, session, self._to_message_objects(messages, session))
        return results

    async def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> SearchResult:
        """检索记忆：语义与关键词双路召回取并集，融合打分后过滤与截断。"""
        limit = DEFAULT_TOP_K if top_k is None else top_k
        floor = DEFAULT_THRESHOLD if threshold is None else threshold
        validate_search_params(floor, limit)
        text = validate_query(query)
        scope = self._build_filters(filters)

        recall_k = max(limit * 4, 60)
        vector = self._embed([text])[0]
        candidates = await self.store.recall(text, vector, scope, recall_k)
        results = self._fuse_and_rank(candidates, floor, limit)
        self._log("记忆检索完成", candidates=len(candidates), hits=len(results))
        return SearchResult(results=results)


    async def update(
        self,
        memory_id: int,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UpdateResult:
        """更新记忆：正文变化才重算向量，元数据合并后覆盖旧值。"""
        if text is None and metadata is None:
            raise DataError("text 与 metadata 至少提供一个")
        current = await self.store.get_memory(memory_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload, previous, changed = self._merge_payload(current.payload, text, metadata)

        if changed:
            vector = self._embed([payload["data"]])[0]
        else:
            # 正文未变：传 None 让后端保留原向量，避免无谓的嵌入调用。
            vector = None

        await self.store.update_memory(memory_id, vector, payload, previous)
        self._log("记忆更新完成", id=memory_id, text_changed=changed)
        return UpdateResult(result=self._to_item(memory_id, payload, None))


    # ---------------- 流程四：删除 ----------------

    async def delete(self, memory_id: int) -> DeleteResult:
        """删除记忆：向量库物理删除，历史表写入 DELETE 审计记录。"""
        current = await self.store.get_memory(memory_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload = current.payload or {}
        await self.store.delete_memory(memory_id, payload.get("data"), payload)
        self._log("记忆删除完成", id=memory_id)
        return DeleteResult(id=memory_id)