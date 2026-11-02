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
 记忆层处理器：装配 embedding、llm、关系库、向量库与可选重排，实现 get、page_query、add、search、update、delete 六条流程。

 分工：本文件只做流程编排与结果组装；存储细节在 stores.py，
 作用域校验在 scope.py，提示词在 prompts.py。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from loong_memories.core.exceptions import ConfigError, DataError, EmbeddingError, LLMError
from loong_memories.embedding.embedder_handler import EmbedderHandler
from loong_memories.llm.llm_handler import LLMHandler
from loong_memories.memory.base import MemoryBase
from loong_memories.memory.models import (
    AddResult,
    DeleteResult,
    GetResult,
    MemoryItem,
    PageQueryResult,
    SearchResult,
    UpdateResult,
)
from loong_memories.memory.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    generate_additive_extraction_prompt,
)
from loong_memories.memory.scope import (
    build_new_metadata,
    session_scope,
    update_metadata_keys,
    validate_query,
    validate_search_params,
)
from loong_memories.memory.stores import (
    AsyncMemoryStore,
    MemoryStore,
    build_payload,
    keyword_text,
    memory_hash,
    now_iso,
    split_payload,
)
from loong_memories.nlp.lemmatization import lemmatize_zh_for_bm25
from loong_memories.reranks.rerank_handler import RerankHandler
from loong_memories.storage.memories.base import VectorStoreResult, MEMORY_TABLE_WITHOUT_VECTOR_COLUMNS
from loong_memories.storage.memories.vector_engine import AsyncVectorEngine, VectorEngine
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine
from loong_memories.storage.message.schema import Message
from loong_memories.util.datetime_util import now_datetime
from loong_memories.util.numberic_util import str_to_int
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
        embedder: EmbedderHandler,
        llm: LLMHandler,
        db: DBEngine,
        vector: VectorEngine,
        reranker: Optional[RerankHandler] = None,
    ) -> None:
        self.embedder = embedder
        self.llm = llm
        self.db = db
        self.vector = vector
        self.reranker = reranker
        self._validate()
        self.store = MemoryStore(vector, db)


    def start(self):
        self.store.start()

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
        """调用大模型抽取事实；模型失败抛 LLMError。"""
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

    @staticmethod
    def _to_payload(result: VectorStoreResult) -> Dict[str, Any]:
        """把存储层的只读结果还原成 payload：核心列直取，metadata 展开回业务键。"""
        payload: Dict[str, Any] = dict(result.metadata or {})
        for key in MEMORY_TABLE_WITHOUT_VECTOR_COLUMNS:
            value = getattr(result, key, None)
            if value is not None:
                payload[key] = value
        return payload


    @staticmethod
    def _to_item(memory_id: int, tenant_id:int, user_id: Optional[int] = None, *, payload: Optional[Dict[str, Any]], score: Optional[float]) -> MemoryItem:
        """payload 转对外条目：核心字段直取，其余归入 metadata。"""
        data = dict(payload or {})
        extra = split_payload(data)
        metadata = extra.get("metadata") or {}
        # split_payload 的 CORE_KEYS 写的是 text_lemmatized，本模块产出 text_lemma，避免其漏进业务元数据。
        metadata.pop("text_lemma", None)
        if metadata:
            extra["metadata"] = metadata
        else:
            extra.pop("metadata", None)
        return MemoryItem(
            id=memory_id,
            tenant_id=tenant_id,
            user_id=user_id,
            memory=str(data.get("text") or ""),
            score=score,
            hash=data.get("hash"),
            created_at_tz=data.get("created_at_tz"),
            updated_at_tz=data.get("updated_at_tz"),
            **extra,
        )


    @staticmethod
    def _to_message_objects(
        messages: List[Dict[str, Any]],
        session: str,
        tenant_id: Any,
        user_id: Any,
    ) -> List[Message]:
        """消息字典转存储对象：会话键与租户、用户由作用域统一填充。"""
        return [
            Message(
                id=next_id(),
                tenant_id=tenant_id,
                user_id=user_id,
                session_scope=session,
                role=message.get("role"),
                content=message.get("content"),
                name=message.get("name"),
                created_at_tz=now_datetime(),
            )
            for message in messages
        ]




    def _recall_existing(self, tenant_id:int, user_id: int | None = None, *, messages: List[Dict[str, Any]], filter: Dict[str, Any]):
        """召回既有记忆：返回既有记忆列表与指纹集合，供提示词与去重共用。"""
        keyword = " ".join(message["content"] for message in messages if message.get("content"))
        vector = self._embed([keyword])[0]
        existing = self.store.recall(
            query=keyword,
            tenant_id=tenant_id,
            vector=vector,
            filter=filter,
            top_k=DEFAULT_CONTEXT_TOP_K,
            user_id=user_id,
        )

        items: List[Dict[str, Any]] = []
        hashes = set()
        for item in existing:
            text = item.get("memory")
            if not isinstance(text, str) or not text:
                continue
            # 提示词要求既存记忆为 [{"id": ..., "text": ...}] 结构。
            items.append({"id": str(item.get("id")), "text": text})
            digest = item.get("hash")
            if isinstance(digest, str) and digest:
                hashes.add(digest)
        return items, hashes


    def _persist_facts(
        self,
        tenant_id: int,
        user_id: int | None = None,
        *,
        facts: List[str],
        new_metadata: Dict[str, Any],
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


        message_objects = self._to_message_objects(messages, session, tenant_id, user_id)

        if not unique:
            # 没有新增也要保存消息，保证会话上下文连续；写入失败只告警，不影响主流程。
            self.store._save_messages(message_objects, session)
            return []

        vectors = self._embed(unique)
        entries: List[Dict[str, Any]] = []
        results: List[MemoryItem] = []
        for text, vector in zip(unique, vectors):
            memory_id = next_id()
            payload = build_payload(text, new_metadata)

            new_payload = payload.copy()
            new_payload.update({"tenant_id": tenant_id, "user_id": user_id})

            entries.append({
                "id": memory_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "vector": vector,
                "payload": new_payload,
            })

            results.append(self._to_item(memory_id, tenant_id, user_id, payload=payload, score=None))

        self.store.insert_memories(entries, session, message_objects)
        return results


    # ---------------- 流程一：读取 ----------------

    def get(self,
            memory_id: Any,
            tenant_id: int,
            user_id: Optional[int] = None
    ) -> GetResult:
        """读取单条记忆：租户必填，用户可选收窄。"""
        current = self.store.get_memory(memory_id, tenant_id, user_id)
        if current is None:
            return GetResult()
        return GetResult(result=self._to_item(memory_id, tenant_id, user_id, payload=self._to_payload(current), score=current.score))


    def page_query(
        self,
        tenant_id: int,
        user_id: Optional[int] = None,
        page_num: int = 0,
        page_size: int = 10,
        *,
        filter: Optional[Dict[str, Any]] = None
    ) -> PageQueryResult:
        """分页查询记忆：在当前作用域内按页码切片。

        存储层没有独立的计数接口，total_size 表示本次抓取窗口内命中的条数。
        """
        if page_num < 0 or page_size <= 0:
            raise DataError("page_num 不能为负，page_size 必须为正")

        rows = self.store.vector_engine.list(
            tenant_id,
            user_id=user_id,
            filter=filter,
            top_k=(page_num + 1) * page_size,
        ) or []
        items = [self._to_item(row.id, tenant_id,user_id,payload=self._to_payload(row), score=row.score) for row in rows]
        start = page_num * page_size
        return PageQueryResult(
            page_num=page_num,
            page_size=page_size,
            total_size=len(items),
            results=items[start:start + page_size],
        )


    # ---------------- 流程二：写入 ----------------

    def add(
        self,
        messages: Any,
        tenant_id: Any,
        user_id: Any = None,
        *,
        agent_id: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AddResult:
        """写入记忆：先由大模型抽取事实，去重后批量落库。"""
        new_metadata, query = build_new_metadata(
            agent_id=agent_id,
            metadata=metadata,
        )
        normalized = self._normalize_messages(messages)

        # 先召回既有记忆：既作为抽取上下文，也用于库内去重。
        existing, existing_hashes = self._recall_existing(tenant_id, user_id, messages=normalized, filter=query)
        session = session_scope(query)
        history = self.store.load_history(session, tenant_id, DEFAULT_HISTORY_MESSAGES, user_id)

        agent_scoped = bool(query.get("agent_id")) and not query.get("user_id")
        facts = self._extract_facts(normalized, existing, history, agent_scoped=agent_scoped)

        recorded = self._persist_facts(tenant_id, user_id, facts=facts, new_metadata=new_metadata, existing_hashes=existing_hashes, session=session, messages=normalized)
        self._log("记忆写入完成", added=len(recorded))
        return AddResult(results=recorded)


    # ---------------- 流程三：检索 ----------------

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
        """检索记忆：语义与关键词双路召回，融合打分后过滤与截断。"""
        limit = DEFAULT_TOP_K if top_k is None else top_k
        floor = DEFAULT_THRESHOLD if threshold is None else threshold
        validate_search_params(floor, limit)
        text = validate_query(query)

        vector = self._embed([text])[0]
        memories = self.store.recall(
            query=text,
            tenant_id=tenant_id,
            vector=vector,
            filter=filter,
            top_k=limit,
            threshold=floor,
            explain=explain,
            user_id=user_id,
        )

        if rerank and self.reranker and memories:
            try:
                memories = self.reranker.rerank(text, memories, limit)
            except Exception as exc:
                logger.error(f"Reranking failed, using original results: {exc}")

        self._log("记忆检索完成", hits=len(memories))
        return SearchResult(results=memories)


    # ---------------- 流程四：更新 ----------------

    def update(
        self,
        memory_id: int,
        tenant_id: int,
        user_id: int | None = None,
        *,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UpdateResult:
        """更新记忆：正文变化才重算向量，元数据合并后覆盖旧值。"""
        if text is None and metadata is None:
            raise DataError("text 与 metadata 至少提供一个")

        current = self.store.get_memory(memory_id, tenant_id, user_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload, previous, changed = self._update_payload(self._to_payload(current), text, metadata)

        if changed:
            vector = self._embed([payload["text"]])[0]
        else:
            # 正文未变：传 None 让后端保留原向量，避免无谓的嵌入调用。
            self._log("记忆未发生变化", id=memory_id, text_changed=changed)
            return UpdateResult(result=self._to_item(memory_id, tenant_id, user_id, payload=payload, score=None))

        self.store.update_memory(memory_id, tenant_id, vector=vector, payload=payload, previous=previous, user_id=user_id)
        self._log("记忆更新完成", id=memory_id, text_changed=changed)
        return UpdateResult(result=self._to_item(memory_id, tenant_id, user_id, payload=payload, score=None))


    def _update_payload(
        self,
        payload: Optional[Dict[str, Any]],
        text: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ):
        """在旧 payload 上合并新值：保留 created_at_tz，刷新 updated_at_tz 与索引列。

        产出键名与 stores.build_payload 完全一致（text、text_lemma、hash、两个 _tz），
        否则向量库按旧键名取值会把正文与词项列写成空值。
        """
        merged = dict(payload or {})
        previous = merged.get("text")
        if metadata:
            merged.update(update_metadata_keys(metadata, merged, context="update"))
        new_text = text if text is not None else previous
        if new_text is None:
            raise DataError("记忆没有正文，无法更新")
        changed = new_text != previous
        if changed:
            merged["text"] = new_text
            merged["text_lemma"] = lemmatize_zh_for_bm25(keyword_text(new_text))
            merged["hash"] = memory_hash(new_text)
        merged.setdefault("created_at_tz", now_iso())
        merged["updated_at_tz"] = now_iso()
        return merged, previous, changed


    # ---------------- 流程五：删除 ----------------

    def delete(self, memory_id: int, tenant_id: int, user_id: int = None) -> DeleteResult:
        """删除记忆：向量库物理删除，历史表写入 DELETE 审计记录。"""
        current = self.store.get_memory(memory_id, tenant_id, user_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload = self._to_payload(current)
        self.store.delete_memory(memory_id, tenant_id, payload.get("text"), payload, user_id)
        self._log("记忆删除完成", id=memory_id)
        return DeleteResult(id=memory_id)


class AsyncMemoryHandler(MemoryHandler):
    """记忆层处理器：六条流程均为异步，存储调用全部 await。"""

    def __init__(
        self,
        embedder: EmbedderHandler,
        llm: LLMHandler,
        db: AsyncDBEngine,
        vector: AsyncVectorEngine,
        reranker: Optional[RerankHandler] = None,
    ) -> None:
        super().__init__(embedder, llm, db, vector, reranker)
        self.store = AsyncMemoryStore(vector, db)


    async def start(self):
        await self.store.start()


    # ---------------- 流程一：读取 ----------------

    async def get(self,
            memory_id: Any,
            tenant_id: int,
            user_id: Optional[int] = None
    ) -> GetResult:
        """读取单条记忆：租户必填，用户可选收窄。"""
        current = await self.store.get_memory(memory_id, tenant_id, user_id)
        if current is None:
            return GetResult()
        return GetResult(result=self._to_item(memory_id, tenant_id, user_id, payload=self._to_payload(current), score=current.score))


    async def page_query(
        self,
        tenant_id: int,
        user_id: Optional[int] = None,
        page_num: int = 0,
        page_size: int = 10,
        *,
        filter: Optional[Dict[str, Any]] = None
    ) -> PageQueryResult:
        """分页查询记忆：在当前作用域内按页码切片。

        存储层没有独立的计数接口，total_size 表示本次抓取窗口内命中的条数。
        """
        if page_num < 0 or page_size <= 0:
            raise DataError("page_num 不能为负，page_size 必须为正")

        rows = await self.store.vector_engine.list(
            tenant_id,
            user_id=user_id,
            filter=filter,
            top_k=(page_num + 1) * page_size,
        ) or []
        items = [self._to_item(row.id, tenant_id,user_id,payload=self._to_payload(row), score=row.score) for row in rows]
        start = page_num * page_size
        return PageQueryResult(
            page_num=page_num,
            page_size=page_size,
            total_size=len(items),
            results=items[start:start + page_size],
        )


    # ---------------- 流程二：写入 ----------------

    async def add(
        self,
        messages: Any,
        tenant_id: Any,
        user_id: Any = None,
        *,
        agent_id: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AddResult:
        """写入记忆：先由大模型抽取事实，去重后批量落库。"""
        new_metadata, query = build_new_metadata(
            agent_id=agent_id,
            metadata=metadata,
        )

        normalized = self._normalize_messages(messages)

        existing, existing_hashes = await self._recall_existing(tenant_id, user_id, messages=normalized, filter=query)
        session = session_scope(query)
        history = await self.store.load_history(session, tenant_id, DEFAULT_HISTORY_MESSAGES, user_id)

        agent_scoped = bool(query.get("agent_id")) and not query.get("user_id")
        facts = self._extract_facts(normalized, existing, history, agent_scoped=agent_scoped)

        recorded = await self._persist_facts(tenant_id, user_id, facts=facts, new_metadata=new_metadata, existing_hashes=existing_hashes, session=session, messages=normalized)
        self._log("记忆写入完成", added=len(recorded))
        return AddResult(results=recorded)


    async def _recall_existing(self, tenant_id:int, user_id: int | None = None, *, messages: List[Dict[str, Any]], filter: Dict[str, Any]):
        """召回既有记忆：返回既有记忆列表与指纹集合，供提示词与去重共用。"""
        keyword = " ".join(message["content"] for message in messages if message.get("content"))
        vector = self._embed([keyword])[0]
        existing = await self.store.recall(
            query=keyword,
            tenant_id=tenant_id,
            vector=vector,
            filter=filter,
            top_k=DEFAULT_CONTEXT_TOP_K,
            user_id=user_id,
        )
        items: List[Dict[str, Any]] = []
        hashes = set()
        for item in existing:
            text = item.get("memory")
            if not isinstance(text, str) or not text:
                continue
            items.append({"id": str(item.get("id")), "text": text})
            digest = item.get("hash")
            if isinstance(digest, str) and digest:
                hashes.add(digest)
        return items, hashes


    async def _persist_facts(
        self,
        tenant_id: int,
        user_id: int | None = None,
        *,
        facts: List[str],
        new_metadata: Dict[str, Any],
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

        message_objects = self._to_message_objects(messages, session, tenant_id, user_id)

        if not unique:
            await self.store._save_messages(message_objects, session)
            return []

        vectors = self._embed(unique)
        entries: List[Dict[str, Any]] = []
        results: List[MemoryItem] = []
        for text, vector in zip(unique, vectors):
            memory_id = next_id()
            payload = build_payload(text, new_metadata)

            new_payload = payload.copy()
            new_payload.update({"tenant_id": tenant_id, "user_id": user_id})

            entries.append({
                "id": memory_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "vector": vector,
                "payload": new_payload,
            })

            results.append(self._to_item(memory_id,tenant_id,user_id,payload=payload, score=None))

        await self.store.insert_memories(entries, session, message_objects)
        return results


    # ---------------- 流程三：检索 ----------------

    async def search(
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
        """检索记忆：语义与关键词双路召回，融合打分后过滤与截断。"""
        limit = DEFAULT_TOP_K if top_k is None else top_k
        floor = DEFAULT_THRESHOLD if threshold is None else threshold
        validate_search_params(floor, limit)
        text = validate_query(query)

        vector = self._embed([text])[0]
        memories = await self.store.recall(
            query=text,
            tenant_id=tenant_id,
            vector=vector,
            filter=filter,
            top_k=limit,
            threshold=floor,
            explain=explain,
            user_id=user_id,
        )

        if rerank and self.reranker and memories:
            try:
                memories = self.reranker.rerank(text, memories, limit)
            except Exception as exc:
                logger.error(f"Reranking failed, using original results: {exc}")

        self._log("记忆检索完成", hits=len(memories))
        return SearchResult(results=memories)


    # ---------------- 流程四：更新 ----------------

    async def update(
        self,
        memory_id: int,
        tenant_id: int,
        user_id: int | None = None,
        *,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UpdateResult:
        """更新记忆：正文变化才重算向量，元数据合并后覆盖旧值。"""
        if text is None and metadata is None:
            raise DataError("text 与 metadata 至少提供一个")

        current = await self.store.get_memory(memory_id, tenant_id, user_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload, previous, changed = self._update_payload(self._to_payload(current), text, metadata)

        if changed:
            vector = self._embed([payload["text"]])[0]
        else:
            # 正文未变：传 None 让后端保留原向量，避免无谓的嵌入调用。
            self._log("记忆未发生变化", id=memory_id, text_changed=changed)
            return UpdateResult(result=self._to_item(memory_id, tenant_id, user_id, payload=payload, score=None))

        await self.store.update_memory(memory_id, tenant_id, vector=vector, payload=payload, previous=previous,user_id=user_id)
        self._log("记忆更新完成", id=memory_id, text_changed=changed)
        return UpdateResult(result=self._to_item(memory_id, tenant_id, user_id, payload=payload, score=None))


    # ---------------- 流程五：删除 ----------------

    async def delete(self, memory_id: int, tenant_id: int, user_id: int = None) -> DeleteResult:
        """删除记忆：向量库物理删除，历史表写入 DELETE 审计记录。"""
        current = await self.store.get_memory(memory_id, tenant_id, user_id)
        if current is None:
            raise DataError("记忆不存在：" + str(memory_id))
        payload = self._to_payload(current)
        await self.store.delete_memory(memory_id, tenant_id, payload.get("text"), payload, user_id)
        self._log("记忆删除完成", id=memory_id)
        return DeleteResult(id=memory_id)

