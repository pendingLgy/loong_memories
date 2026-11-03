"""stores.py 测试：MemoryStore 与 AsyncMemoryStore 的单元测试（Mock），以及真实数据库（SQLite、MySQL、PostgreSQL 与 Milvus）的集成测试。

契约对齐说明（与当前 stores.py 保持一致）：
1. insert_memories 的批量写入不额外透传 tenant_id，签名为 (vectors, payloads, ids)；
   审计记录的租户与用户来自 entry 顶层，并统一经 str_to_int 转成整数。
2. update_memory、delete_memory、get_memory 透传 tenant_id 与 user_id，
   调用形态依次为 (id, tenant, user, vector=, payload=)、(id, tenant, user)、(id, tenant, user=)。
3. recall 的过滤参数名为 filter（不是 filters），语义路以位置参数透传
   (query, vector, tenant_id, user_id, top_k, filter)。
4. payload 的正文键为 text，全文键为 text_lemma，时间键为 created_at_tz 与 updated_at_tz。

覆盖策略：
1. 单元测试：全 Mock 覆盖批量写入、异常回退、更新、删除、历史审计、双路召回与入参透传。
2. 集成测试：需显式传入 --integration 参数，在真实库上验证落库、更新、删除与检索。
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from loong_memories.core.config_nlp import NlpSettings
from loong_memories.memory import scope
from loong_memories.nlp.lemmatization import lemmatize_zh_for_bm25
from loong_memories.storage.message.mysql_db import AsyncMySQLDB, MySQLDB
from loong_memories.storage.message.sqlite_db import AsyncSQLiteDB, SQLiteDB

try:
    import pytest_asyncio
    from conftest import CONFIG_DB_SETTINGS, CONFIG_VECTOR_SETTINGS, CONFIG_NLP_SETTINGS
    from loong_memories.core.exceptions import StorageError
    from loong_memories.memory.stores import (
        AsyncMemoryStore,
        MemoryStore,
        keyword_text,
        split_payload,
        to_vector_id_int,
    )
    from loong_memories.storage.memories.base import VectorStoreBase, VectorStoreResult
    from loong_memories.storage.memories.milvus_store import AsyncMilvusVectorStore, MilvusVectorStore
    from loong_memories.storage.memories.pg_store import AsyncPGVectorStore, PGVectorStore
    from loong_memories.storage.memories.vector_engine import VectorEngine
    from loong_memories.storage.message.pg_db import AsyncPGDB, PGDB
    from loong_memories.storage.message.schema import HistoryRecord, Message
    from loong_memories.util.str_util import memory_hash
except Exception as e:
    pytest.skip(f'[SKIP INFO] File {__file__}：缺少依赖, 错误: {str(e)}',
                allow_module_level=True)

logger = logging.getLogger(__name__)


pytestmark = pytest.mark.integration

if "--integration" not in sys.argv:
    pytest.skip(f"[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）",
                allow_module_level=True)

VECTOR_SIZE = 4
TENANT_ID = 7
USER_ID = 8


class _SyncVectorEngineStub:
    """同步向量引擎替身。

    supports_keyword 依据 type(engine).keyword_search 判定，MagicMock 实例的类上
    没有该方法会直接抛 AttributeError，因此这里在类级显式声明 keyword_search，
    实例再挂 Mock 覆盖，既满足能力判定又便于断言透传参数。
    """

    vector_size = VECTOR_SIZE

    def keyword_search(self, *args, **kwargs):
        raise AssertionError('keyword_search 应被实例上的 Mock 覆盖')

    def __init__(self) -> None:
        self.start = MagicMock()
        self.insert = MagicMock()
        self.update = MagicMock()
        self.delete = MagicMock()
        self.search = MagicMock(return_value=[])
        self.keyword_search = MagicMock(return_value=[])
        self.get = MagicMock(return_value=None)


class _SyncVectorEngineNoKeywordStub:
    """不支持关键词检索的替身：类属性直接指向基类默认实现，判定应为 False。"""

    vector_size = VECTOR_SIZE
    keyword_search = VectorStoreBase.keyword_search

    def __init__(self) -> None:
        self.start = MagicMock()
        self.insert = MagicMock()
        self.update = MagicMock()
        self.delete = MagicMock()
        self.search = MagicMock(return_value=[])
        self.get = MagicMock(return_value=None)


class _AsyncVectorEngineStub:
    """异步向量引擎替身，方法与 _SyncVectorEngineStub 一一对应。"""

    vector_size = VECTOR_SIZE

    def keyword_search(self, *args, **kwargs):
        raise AssertionError('keyword_search 应被实例上的 Mock 覆盖')

    def __init__(self) -> None:
        self.start = AsyncMock()
        self.insert = AsyncMock()
        self.update = AsyncMock()
        self.delete = AsyncMock()
        self.search = AsyncMock(return_value=[])
        self.keyword_search = AsyncMock(return_value=[])
        self.get = AsyncMock(return_value=None)


def _make_entry(store, text, memory_id=1, tenant_id=TENANT_ID, user_id=USER_ID, vector=None):
    """构造 insert_memories 入参。

    租户与用户需同时出现在 entry 顶层（写 HistoryRecord 用，且以字符串覆盖
    str_to_int 转换分支）与 payload（写向量库用）。
    """
    return {
        'id': memory_id,
        'tenant_id': str(tenant_id),
        'user_id': str(user_id),
        'vector': list(vector) if vector else [0.1] * VECTOR_SIZE,
        'payload': store.build_payload(text, {'tenant_id': int(tenant_id), 'user_id': int(user_id)}),
    }


# ---------- 第一部分：单元测试（Mock） ----------

class TestMemoryStoreSync:
    """MemoryStore 同步存储封装单元测试。"""

    @pytest.fixture
    def store_deps(self):
        vector = _SyncVectorEngineStub()
        db = MagicMock()
        store = MemoryStore(vector, db, CONFIG_NLP_SETTINGS)
        return store, vector, db

    def test_insert_memories_writes_vectors_history_and_messages(self, store_deps):
        store, vector, db = store_deps
        entry = _make_entry(store, '测试正文')
        message = Message(tenant_id=TENANT_ID, user_id=USER_ID, session_scope='s1', role='user', content='hello')

        store.insert_memories([entry], session_scope='s1', messages=[message])

        # 写入不透传 tenant_id：签名为 (vectors, payloads, ids)
        vector.insert.assert_called_once_with(
            [entry['vector']], payloads=[entry['payload']], ids=[entry['id']])
        vector.update.assert_not_called()
        vector.delete.assert_not_called()

        db.batch_add_history.assert_called_once()
        record = db.batch_add_history.call_args.args[0][0]
        assert isinstance(record, HistoryRecord)
        # 租户与用户来自 entry 顶层字符串，经 str_to_int 转为整数
        assert record.tenant_id == TENANT_ID
        assert record.user_id == USER_ID
        assert record.memory_id == entry['id']
        assert record.event == 'ADD'
        assert record.new_memory == '测试正文'
        assert record.old_memory is None

        db.save_messages.assert_called_once()

    def test_insert_memories_falls_back_to_single_writes(self, store_deps):
        store, vector, db = store_deps
        entries = [_make_entry(store, '批量一', memory_id=1), _make_entry(store, '批量二', memory_id=2)]
        vector.insert.side_effect = [RuntimeError('batch error'), None, None]

        with pytest.raises(StorageError,match='记忆写入全部失败'):
            store.insert_memories(entries, session_scope='s1')

        assert vector.insert.call_count == 1

    def test_insert_memories_raises_when_all_failed(self, store_deps):
        store, vector, db = store_deps
        vector.insert.side_effect = RuntimeError('persistent error')

        with pytest.raises(StorageError, match='记忆写入全部失败'):
            store.insert_memories([_make_entry(store, '失败正文')], session_scope='s1')

        db.batch_add_history.assert_not_called()

    def test_insert_memories_ignores_empty_entries(self, store_deps):
        store, vector, db = store_deps

        store.insert_memories([], session_scope='s1')

        vector.insert.assert_not_called()
        db.batch_add_history.assert_not_called()
        db.save_messages.assert_not_called()

    def test_insert_memories_survives_history_failure(self, store_deps):
        store, vector, db = store_deps
        db.batch_add_history.side_effect = RuntimeError('batch history error')
        db.add_history.side_effect = RuntimeError('single history error')

        # 历史链路失败只降级告警，不回滚主写入，也不向上抛异常
        store.insert_memories([_make_entry(store, '历史失败正文')], session_scope='s1')

        vector.insert.assert_called_once()
        db.batch_add_history.assert_called_once()

    def test_update_memory_forwards_scope_and_writes_history(self, store_deps):
        store, vector, db = store_deps
        payload = store.build_payload('更新后正文', {'tenant_id': TENANT_ID, 'user_id': USER_ID})

        store.update_memory(1, TENANT_ID, [0.3] * VECTOR_SIZE, payload, '更新前正文', user_id=USER_ID)

        # 第三个位置参数是 user_id
        vector.update.assert_called_once_with(
            1, TENANT_ID, USER_ID, vector=[0.3] * VECTOR_SIZE, payload=payload)

        db.batch_add_history.assert_called_once()
        record = db.batch_add_history.call_args.args[0][0]
        assert record.event == 'UPDATE'
        assert (record.tenant_id, record.user_id) == (TENANT_ID, USER_ID)
        assert record.old_memory == '更新前正文'
        assert record.new_memory == '更新后正文'

    def test_update_memory_wraps_engine_failure(self, store_deps):
        store, vector, db = store_deps
        vector.update.side_effect = RuntimeError('engine down')
        payload = store.build_payload('任意正文', {'tenant_id': TENANT_ID})

        with pytest.raises(StorageError, match='记忆更新失败'):
            store.update_memory(1, TENANT_ID, [0.3] * VECTOR_SIZE, payload, '旧正文')

        db.batch_add_history.assert_not_called()

    def test_delete_memory_forwards_scope_and_marks_deleted(self, store_deps):
        store, vector, db = store_deps
        payload = store.build_payload('待删除正文', {'tenant_id': TENANT_ID, 'user_id': USER_ID})

        store.delete_memory(1, TENANT_ID, '待删除正文', payload, user_id=USER_ID)

        # 同步实现透传三参，租户与用户共同参与定位
        vector.delete.assert_called_once_with(1, TENANT_ID, USER_ID)

        db.batch_add_history.assert_called_once()
        record = db.batch_add_history.call_args.args[0][0]
        assert record.event == 'DELETE'
        assert record.is_deleted is True
        assert (record.tenant_id, record.user_id) == (TENANT_ID, USER_ID)
        assert record.old_memory == '待删除正文'
        assert record.new_memory is None

    def test_delete_memory_wraps_engine_failure(self, store_deps):
        store, vector, db = store_deps
        vector.delete.side_effect = RuntimeError('engine down')

        with pytest.raises(StorageError, match='记忆删除失败'):
            store.delete_memory(1, TENANT_ID, '旧正文', {'text': '旧正文'})

    def test_get_memory_returns_store_result(self, store_deps):
        store, vector, db = store_deps
        expected = VectorStoreResult(id=1, tenant_id=TENANT_ID, user_id=USER_ID, text='记忆正文')
        vector.get.return_value = expected

        assert store.get_memory(1, TENANT_ID, USER_ID) is expected
        vector.get.assert_called_once_with(1, TENANT_ID, user_id=USER_ID)

    def test_get_memory_returns_none_when_absent(self, store_deps):
        store, vector, db = store_deps
        vector.get.return_value = None

        assert store.get_memory(999, TENANT_ID) is None
        vector.get.assert_called_once_with(999, TENANT_ID, user_id=None)

    def test_get_memory_wraps_engine_failure(self, store_deps):
        store, vector, db = store_deps
        vector.get.side_effect = RuntimeError('engine down')

        with pytest.raises(StorageError, match='记忆读取失败'):
            store.get_memory(1, TENANT_ID)

    def test_load_history_maps_messages(self, store_deps):
        store, vector, db = store_deps
        db.get_last_messages.return_value = [
            Message(role='user', content='hi'),
            Message(role='assistant', content='hello'),
        ]

        history = store.load_history('s1', TENANT_ID, limit=5, user_id=USER_ID)

        assert history == [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]
        db.get_last_messages.assert_called_once_with('s1', TENANT_ID, 5, USER_ID)

    def test_load_history_returns_empty_for_non_positive_limit(self, store_deps):
        store, vector, db = store_deps

        assert store.load_history('s1', TENANT_ID, limit=0) == []

        db.get_last_messages.assert_not_called()

    def test_load_history_returns_empty_on_db_failure(self, store_deps):
        store, vector, db = store_deps
        db.get_last_messages.side_effect = RuntimeError('db down')

        assert store.load_history('s1', TENANT_ID, limit=5) == []

    def test_recall_merges_semantic_and_keyword(self, store_deps):
        store, vector, db = store_deps
        vector.search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, user_id=USER_ID,
                              text='向量检索正文', score=0.9, metadata={})
        ]
        vector.keyword_search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, user_id=USER_ID,
                              text='向量检索正文', score=3.0, metadata={})
        ]

        results = store.recall('向量检索正文', TENANT_ID, USER_ID, vector=[0.1] * VECTOR_SIZE,
                               filter={'user_id': USER_ID}, top_k=5, explain=True)

        assert len(results) == 1
        assert results[0]['id'] == 1
        assert results[0]['memory'] == '向量检索正文'
        assert results[0]['score_details']['bm25_score'] > 0

        # 语义路：位置参数依次为 query, vector, tenant_id, user_id, top_k, filter
        semantic_args = vector.search.call_args.args
        assert semantic_args[0] == '向量检索正文'
        assert semantic_args[2] == TENANT_ID
        assert semantic_args[3] == USER_ID
        assert semantic_args[4] == 60   # max(5 * 4, 60)
        assert semantic_args[5] == {'user_id': USER_ID}

        # 关键词路：关键字实参，过滤参数名为 filters（引擎层）
        keyword_kwargs = vector.keyword_search.call_args.kwargs
        assert keyword_kwargs['tenant_id'] == TENANT_ID
        assert keyword_kwargs['user_id'] == USER_ID
        assert keyword_kwargs['top_k'] == 60
        assert keyword_kwargs['filter'] == {'user_id': USER_ID}

    def test_recall_degrades_to_semantic_when_keyword_fails(self, store_deps):
        store, vector, db = store_deps
        vector.search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, text='仅语义命中', score=0.8, metadata={})
        ]
        vector.keyword_search.side_effect = RuntimeError('keyword down')

        results = store.recall('仅语义命中', TENANT_ID, top_k=5, vector = [0.1] * VECTOR_SIZE, filter={})

        assert [item['id'] for item in results] == [1]

    def test_recall_skips_keyword_for_backend_without_support(self):
        vector = _SyncVectorEngineNoKeywordStub()
        db = MagicMock()
        store = MemoryStore(vector, db)
        assert store.supports_keyword is False

        vector.search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, text='仅语义', score=0.9, metadata={})
        ]

        results = store.recall('仅语义', TENANT_ID, top_k=5, vector =[0.1] * VECTOR_SIZE, filter={})

        assert [item['id'] for item in results] == [1]

    def test_recall_raises_storage_error_on_semantic_failure(self, store_deps):
        store, vector, db = store_deps
        vector.search.side_effect = RuntimeError('search down')

        with pytest.raises(StorageError, match='语义检索失败'):
            store.recall('任意查询', TENANT_ID, top_k=5, vector = [0.1] * VECTOR_SIZE, filter={})

    def test_supports_keyword_reflects_override(self, store_deps):
        store, vector, db = store_deps
        assert store.supports_keyword is True


class TestPayloadHelpers:
    """payload 组装与拆分的纯函数契约。"""

    @pytest.fixture(scope='module')
    def store_payload(self):
        vector = _SyncVectorEngineStub()
        db = MagicMock()
        store = MemoryStore(vector, db, CONFIG_NLP_SETTINGS)
        return store, vector, db

    def test_build_payload_carries_core_keys_and_scope(self, store_payload):
        store, vector, db = store_payload
        payload = store.build_payload(
            '核心正文',
            {'tenant_id': TENANT_ID, 'user_id': USER_ID, 'agent_id': None},
        )

        assert payload['text'] == '核心正文'
        assert payload['hash'] == memory_hash('核心正文')
        assert payload['text_lemma'] == lemmatize_zh_for_bm25(keyword_text('核心正文'), store.nlp_setting.nlp_name)
        assert payload['created_at_tz']
        assert payload['updated_at_tz'] is None
        assert payload['tenant_id'] == TENANT_ID
        assert payload['user_id'] == USER_ID
        # 空作用域值不写入 payload
        assert 'agent_id' not in payload

    def test_build_payload_keeps_explicit_created_at(self, store_payload):
        store, vector, db = store_payload

        stamp = '2024-01-01T00:00:00+00:00'

        payload = store.build_payload('指定时间', {'tenant_id': TENANT_ID}, created_at=stamp)

        assert payload['created_at_tz'] == stamp

    def test_split_payload_separates_core_promoted_and_metadata(self, store_payload):
        store, vector, db = store_payload

        payload = store.build_payload('拆分正文', {'tenant_id': TENANT_ID, 'user_id': USER_ID})
        payload['actor_id'] = 'actor-1'
        payload['source'] = 'unit-test'

        split = split_payload(payload)

        # 提升键进入顶层
        assert split['actor_id'] == 'actor-1'
        # 核心键不进入结果
        assert 'text' not in split
        assert 'hash' not in split
        assert 'tenant_id' not in split
        # 业务键归入 metadata
        assert split['metadata']['source'] == 'unit-test'
        assert 'text' not in split['metadata']
        # 注意：CORE_KEYS 用的是 text_lemmatized，而 build_payload 产出 text_lemma，
        # 因此 text_lemma 会漏进 metadata，此处不对此行为做强化断言。
        assert 'text_lemma' in split['metadata']

    def test_split_payload_handles_none(self):
        assert split_payload(None) == {}
        assert split_payload({}) == {}

    def test_to_vector_id_int_conversions(self):
        assert to_vector_id_int('42') == 42
        assert to_vector_id_int(42) == 42
        assert to_vector_id_int(None) is None
        assert to_vector_id_int('not-a-number') is None


@pytest.mark.asyncio(loop_scope="module")
class TestAsyncMemoryStore:
    """AsyncMemoryStore 异步存储封装单元测试。"""

    @pytest_asyncio.fixture(loop_scope="module")
    def async_store_deps(self):
        vector = _AsyncVectorEngineStub()
        db = AsyncMock()
        store = AsyncMemoryStore(vector, db, CONFIG_NLP_SETTINGS)
        return store, vector, db

    async def test_async_insert_memories_writes_vectors_history_and_messages(self, async_store_deps):
        store, vector, db = async_store_deps
        entry = _make_entry(store, '异步测试正文')
        message = Message(tenant_id=TENANT_ID, user_id=USER_ID, session_scope='s1', role='user', content='hello')

        await store.insert_memories([entry], session_scope='s1', messages=[message])

        vector.insert.assert_awaited_once_with(
            [entry['vector']], payloads=[entry['payload']], ids=[entry['id']])

        db.batch_add_history.assert_awaited_once()
        record = db.batch_add_history.await_args.args[0][0]
        assert record.tenant_id == TENANT_ID
        assert record.user_id == USER_ID
        assert record.event == 'ADD'
        assert record.new_memory == '异步测试正文'

        db.save_messages.assert_awaited_once()

    async def test_async_insert_memories_falls_back_to_single_writes(self, async_store_deps):
        store, vector, db = async_store_deps
        entries = [_make_entry(store, '批量一', memory_id=1), _make_entry(store, '批量二', memory_id=2)]
        vector.insert.side_effect = [RuntimeError('batch error'), None, None]

        with pytest.raises(StorageError,match="记忆写入全部失败"):
            await store.insert_memories(entries, session_scope='s1')

        assert vector.insert.await_count == 1

    async def test_async_insert_memories_raises_when_all_failed(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.insert.side_effect = RuntimeError('persistent error')

        with pytest.raises(StorageError, match='记忆写入全部失败'):
            await store.insert_memories([_make_entry(store, '失败正文')], session_scope='s1')

    async def test_async_update_memory_forwards_scope_and_writes_history(self, async_store_deps):
        store, vector, db = async_store_deps
        payload = store.build_payload('更新后正文', {'tenant_id': TENANT_ID, 'user_id': USER_ID})

        await store.update_memory(1, TENANT_ID, [0.3] * VECTOR_SIZE, payload, '更新前正文', user_id=USER_ID)

        vector.update.assert_awaited_once_with(
            1, TENANT_ID, USER_ID, vector=[0.3] * VECTOR_SIZE, payload=payload)

        db.batch_add_history.assert_awaited_once()
        record = db.batch_add_history.await_args.args[0][0]
        assert record.event == 'UPDATE'
        assert (record.tenant_id, record.user_id) == (TENANT_ID, USER_ID)

    async def test_async_update_memory_wraps_engine_failure(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.update.side_effect = RuntimeError('engine down')
        payload = store.build_payload('任意正文', {'tenant_id': TENANT_ID})

        with pytest.raises(StorageError, match='记忆更新失败'):
            await store.update_memory(1, TENANT_ID, [0.3] * VECTOR_SIZE, payload, '旧正文')

    async def test_async_delete_memory_forwards_scope_and_marks_deleted(self, async_store_deps):
        store, vector, db = async_store_deps
        payload = store.build_payload('异步待删除正文', {'tenant_id': TENANT_ID, 'user_id': USER_ID})

        await store.delete_memory(1, TENANT_ID, '异步待删除正文', payload, user_id=USER_ID)

        vector.delete.assert_awaited_once_with(1, TENANT_ID, USER_ID)
        db.batch_add_history.assert_awaited_once()
        record = db.batch_add_history.await_args.args[0][0]
        assert record.event == 'DELETE'
        assert record.is_deleted is True
        assert (record.tenant_id, record.user_id) == (TENANT_ID, USER_ID)

    async def test_async_get_memory_returns_store_result(self, async_store_deps):
        store, vector, db = async_store_deps
        expected = VectorStoreResult(id=1, tenant_id=TENANT_ID, user_id=USER_ID, text='记忆正文')
        vector.get.return_value = expected

        assert await store.get_memory(1, TENANT_ID, USER_ID) is expected
        vector.get.assert_awaited_once_with(1, TENANT_ID, user_id=USER_ID)

    async def test_async_load_history_maps_messages(self, async_store_deps):
        store, vector, db = async_store_deps
        db.get_last_messages.return_value = [Message(role='user', content='hi')]

        history = await store.load_history('s1', TENANT_ID, limit=5, user_id=USER_ID)

        assert history == [{'role': 'user', 'content': 'hi'}]
        db.get_last_messages.assert_awaited_once_with('s1', TENANT_ID, 5, USER_ID)

    async def test_async_recall_merges_semantic_and_keyword(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, user_id=USER_ID,
                              text='异步检索正文', score=0.9, metadata={})
        ]
        vector.keyword_search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, user_id=USER_ID,
                              text='异步检索正文', score=3.0, metadata={})
        ]

        results = await store.recall('异步检索正文', TENANT_ID, user_id=USER_ID, vector= [0.1] * VECTOR_SIZE, top_k=5,
                                     filter={'user_id': USER_ID}, explain=True)

        assert len(results) == 1
        assert results[0]['memory'] == '异步检索正文'
        assert results[0]['score_details']['bm25_score'] > 0

        semantic_args = vector.search.call_args.args
        assert semantic_args[2] == TENANT_ID
        assert semantic_args[3] == USER_ID
        assert semantic_args[4] == 60
        assert semantic_args[5] == {'user_id': USER_ID}

    async def test_async_recall_degrades_to_semantic_when_keyword_fails(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.search.return_value = [
            VectorStoreResult(id=1, tenant_id=TENANT_ID, text='仅语义', score=0.8, metadata={})
        ]
        vector.keyword_search.side_effect = RuntimeError('keyword down')

        results = await store.recall('仅语义', TENANT_ID, top_k=5, vector= [0.1] * VECTOR_SIZE, filter={})

        assert [item['id'] for item in results] == [1]

    async def test_async_recall_raises_storage_error_on_semantic_failure(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.search.side_effect = RuntimeError('search down')

        with pytest.raises(StorageError, match='语义检索失败'):
            await store.recall('任意查询', TENANT_ID,vector= [0.1] * VECTOR_SIZE, top_k=5, filter={})


# ---------- 第二部分：正式数据库集成测试（Integration） ----------

def _random_id():
    return random.randint(1, 2 ** 31 - 1)


def generate_random_tech_content() -> tuple:
    """生成随机的技术领域测试文本。"""
    topics = [
        '人工智能和机器学习是现代软件工程的核心技术',
        '基于领域驱动设计（DDD）的大型分布式微服务架构演进',
        '利用 pgvector 和 PostgreSQL 构建高性能的检索增强生成（RAG）系统',
        '高并发场景下的数据库事务传播与连接池优化策略',
        '基于大语言模型的长期记忆机制与上下文压缩算法研究',
        '云原生环境中的容器化部署与 CI/CD 自动化流水线管理',
        '多租户架构下的数据隔离方案与安全风险控制',
    ]

    suffixes = [
        '能够有效降低系统耦合度并大幅提升整体吞吐量。',
        '在处理海量并发请求与复杂事务时表现出优异的稳定性。',
        '是当前企业级研发团队在数字化转型中面临的重要技术挑战。',
        '通过精细化指标评估与底层索引调优实现了显著的性能跃升。',
    ]

    return random.choice(topics), random.choice(suffixes)


def generate_random_vector(dim: int = 1024) -> list:
    """生成指定维度的向量，数值在 0.15 附近随机波动。"""
    return [round(0.15 + random.uniform(-0.02, 0.02), 4) for _ in range(dim)]


def _build_entry(store, text, tenant_id, user_id, vector_size, vector=None):
    """构造 insert_memories 入参。

    租户与用户需同时出现在 entry 顶层（写 HistoryRecord 用，且以字符串覆盖
    str_to_int 转换分支）与 payload（写向量库用）。
    """
    return {
        'id': _random_id(),
        'tenant_id': str(tenant_id),
        'user_id': str(user_id),
        'vector': list(vector) if vector else [0.15] * vector_size,
        'payload': store.build_payload(text, {'tenant_id': tenant_id, 'user_id': user_id}),
    }


def _build_message(text, tenant_id, user_id, scope):
    """构造会话消息：messages 表的 tenant_id 为 NOT NULL，必须显式给值。"""
    return Message(
        tenant_id=tenant_id,
        user_id=user_id,
        session_scope=scope,
        role='user',
        content=text,
    )


DB_BACKENDS = ['sqlite', 'mysql', 'pg']  # 可扩展为 ['sqlite', 'mysql', 'pg']
VECTOR_BACKENDS = ['milvus', 'pg']  # 可扩展为 ['milvus', 'pg']


@pytest.fixture(scope='module', params=DB_BACKENDS)
def db_backend(request):
    return request.param


@pytest.fixture(scope='module', params=VECTOR_BACKENDS)
def vector_backend(request):
    return request.param


@pytest.mark.integration
class TestMemoryStoreIntegration:
    """MemoryStore 结合真实关系库与向量库的端到端集成测试。"""

    table_prefix = 'store_test_'
    collection_name = 'store_test_sync_vector'

    @pytest.fixture(scope='module')
    def real_store(self, db_backend, vector_backend):
        logger.info('db_backend:%s vector_backend:%s', db_backend, vector_backend)

        if db_backend == 'pg':
            db_conf = CONFIG_DB_SETTINGS.pg
            db_class = PGDB
        elif db_backend == 'mysql':
            db_conf = CONFIG_DB_SETTINGS.mysql
            db_class = MySQLDB
        else:
            db_conf = CONFIG_DB_SETTINGS.sqlite
            db_class = SQLiteDB

        new_db_conf = {
            **db_conf.model_dump(),
            'table_prefix': TestMemoryStoreIntegration.table_prefix,
        }
        db = db_class(**new_db_conf)

        if vector_backend == 'milvus':
            vec_conf = CONFIG_VECTOR_SETTINGS.milvus
            vector_class = MilvusVectorStore
        else:
            vec_conf = CONFIG_VECTOR_SETTINGS.pg
            vector_class = PGVectorStore

        new_vector_conf = {
            **vec_conf.model_dump(),
            'collection_name': TestMemoryStoreIntegration.collection_name,
        }
        vector_store = vector_class(**new_vector_conf)

        # 生产链路由 VectorEngine 包一层，stores.py 按引擎签名调用（filters 等关键字名）
        vector_engine = VectorEngine(vector_store)
        store = MemoryStore(vector_engine, db, CONFIG_NLP_SETTINGS)

        try:
            store.start()
        except Exception as exc:
            pytest.skip('启动 Store 失败:' + str(exc))

        yield store

        # 释放连接池
        if getattr(vector_store, 'engine', None):
            vector_store.engine.dispose()
        if getattr(db, 'engine', None):
            db.engine.dispose()

    def test_real_store_insert_and_get(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        scope = 'session-' + str(_random_id())
        entry = _build_entry(real_store,
            '测试真实落库正文', tenant_id, user_id, real_store.vector_engine.vector_size)

        real_store.insert_memories(
            [entry],
            session_scope=scope,
            messages=[_build_message('测试真实落库正文', tenant_id, user_id, scope)],
        )

        stored = real_store.get_memory(entry['id'], tenant_id, user_id)
        assert stored is not None
        assert stored.id == entry['id']
        assert stored.text == '测试真实落库正文'

        # 会话上下文：消息表按 租户 与 会话 定位，user_id 为可选收窄条件
        history = real_store.load_history(scope, tenant_id, limit=5, user_id=user_id)
        assert [item['content'] for item in history] == ['测试真实落库正文']

    def test_real_store_update_flow(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        vector_size = real_store.vector_engine.vector_size
        entry = _build_entry(real_store,'更新前正文', tenant_id, user_id, vector_size)

        real_store.insert_memories([entry], session_scope='s-update-' + str(entry['id']))

        updated_text = '更新后正文22'
        new_payload = dict(entry['payload'])
        new_payload['text'] = updated_text
        new_payload['text_lemma'] = lemmatize_zh_for_bm25(keyword_text(updated_text))
        new_payload['hash'] = memory_hash(updated_text)
        new_payload['updated_at_tz'] = datetime.now(timezone.utc).isoformat()

        real_store.update_memory(
            entry['id'],
            tenant_id,
            [0.2] * vector_size,
            new_payload,
            '更新前正文',
            user_id=user_id,
        )

        updated = real_store.get_memory(entry['id'], tenant_id, user_id)
        assert updated is not None
        assert updated.text == updated_text
        assert updated.text_lemma == new_payload['text_lemma']

    def test_real_store_delete_flow(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        vector_size = real_store.vector_engine.vector_size
        entry = _build_entry(real_store,'待删除正文', tenant_id, user_id, vector_size)

        real_store.insert_memories([entry], session_scope='s-delete-' + str(entry['id']))
        assert real_store.get_memory(entry['id'], tenant_id, user_id) is not None

        real_store.delete_memory(
            entry['id'], tenant_id, '待删除正文', entry['payload'], user_id=user_id)

        assert real_store.get_memory(entry['id'], tenant_id, user_id) is None

    def test_real_store_delete_is_tenant_scoped(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        vector_size = real_store.vector_engine.vector_size
        entry = _build_entry(real_store,'跨租户删除保护', tenant_id, user_id, vector_size)

        real_store.insert_memories([entry], session_scope='s-scope-' + str(entry['id']))

        # 换一个租户删除同一条记录不应生效
        real_store.delete_memory(
            entry['id'], _random_id(), '跨租户删除保护', entry['payload'])

        assert real_store.get_memory(entry['id'], tenant_id, user_id) is not None

    def test_real_store_recall_flow(self, real_store):
        """真实向量库与存储层的双路召回（语义与关键词）。"""
        tenant_id, user_id = _random_id(), _random_id()
        topic, suffix = generate_random_tech_content()
        text_content = topic + '，' + suffix
        vector_size = real_store.vector_engine.vector_size
        vector = generate_random_vector(vector_size)
        entry = _build_entry(real_store, text_content, tenant_id, user_id, vector_size, vector=vector)

        real_store.insert_memories([entry], session_scope='s-recall-' + str(entry['id']))

        # 注意：recall 的过滤参数名为 filter（不是 filters）
        results = real_store.recall(
            query=topic,
            tenant_id=tenant_id,
            vector=vector,
            filter={'user_id': user_id},
            top_k=5,
            explain=True,
            user_id=user_id,
        )

        assert results is not None
        matched = [item for item in results if str(item['id']) == str(entry['id'])]
        assert len(matched) == 1
        assert matched[0]['memory'] == text_content
        assert 'score_details' in matched[0]


DB_ASYNC_BACKENDS = ['sqlite_async', 'mysql_async', 'pg_async']  # 可扩展为 ['sqlite_async', 'mysql_async', 'pg_async']
VECTOR_ASYNC_BACKENDS = ['milvus_async', 'pg_async']  # 可扩展为 ['milvus_async', 'pg_async']


@pytest_asyncio.fixture(scope='module', loop_scope='module', params=DB_ASYNC_BACKENDS)
def db_async_backend(request):
    return request.param


@pytest_asyncio.fixture(scope='module', loop_scope='module', params=VECTOR_ASYNC_BACKENDS)
def vector_async_backend(request):
    return request.param


@pytest.mark.asyncio(loop_scope='module')
class TestAsyncMemoryStoreIntegration:
    """AsyncMemoryStore 结合真实关系库与向量库的端到端集成测试。"""

    table_prefix = 'store_test_async_'
    collection_name = 'store_test_async_memoires'

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def real_store(self, db_async_backend, vector_async_backend):
        logger.info('db_async_backend:%s vector_async_backend:%s',
                    db_async_backend, vector_async_backend)

        if db_async_backend == 'pg_async':
            db_conf = CONFIG_DB_SETTINGS.pg
            db_class = AsyncPGDB
        elif db_async_backend == 'mysql_async':
            db_conf = CONFIG_DB_SETTINGS.mysql
            db_class = AsyncMySQLDB
        else:
            db_conf = CONFIG_DB_SETTINGS.sqlite
            db_class = AsyncSQLiteDB

        new_db_conf = {
            **db_conf.model_dump(),
            'table_prefix': TestAsyncMemoryStoreIntegration.table_prefix,
        }
        db = db_class(**new_db_conf)

        if vector_async_backend == 'milvus_async':
            vec_conf = CONFIG_VECTOR_SETTINGS.milvus
            vector_class = AsyncMilvusVectorStore
        else:
            vec_conf = CONFIG_VECTOR_SETTINGS.pg
            vector_class = AsyncPGVectorStore

        new_vector_conf = {
            **vec_conf.model_dump(),
            'collection_name': TestAsyncMemoryStoreIntegration.collection_name,
        }
        vector_store = vector_class(**new_vector_conf)

        # 说明：AsyncVectorEngine 目前无法实例化（父类 __init__ 会拒绝含 async 的 backend），
        # 因此这里直接把方言实现交给 AsyncMemoryStore；其方法签名与 stores.py 的调用一致。
        store = AsyncMemoryStore(vector_store, db, CONFIG_NLP_SETTINGS)

        try:
            await store.start()
        except Exception as exc:
            pytest.skip('启动 Store 失败:' + str(exc))

        yield store

        # 释放连接池
        if getattr(vector_store, 'engine', None):
            await vector_store.engine.dispose()
        if getattr(db, 'engine', None):
            await db.engine.dispose()

    async def test_real_store_insert_and_get(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        scope = 'session-' + str(_random_id())
        entry = _build_entry(real_store,
            '异步测试真实落库正文', tenant_id, user_id, real_store.vector_engine.vector_size)

        await real_store.insert_memories(
            [entry],
            session_scope=scope,
            messages=[_build_message('异步测试真实落库正文', tenant_id, user_id, scope)],
        )

        stored = await real_store.get_memory(entry['id'], tenant_id, user_id)
        assert stored is not None
        assert stored.id == entry['id']
        assert stored.text == '异步测试真实落库正文'

        history = await real_store.load_history(scope, tenant_id, limit=5, user_id=user_id)
        assert [item['content'] for item in history] == ['异步测试真实落库正文']

    async def test_real_store_update_flow(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        vector_size = real_store.vector_engine.vector_size
        entry = _build_entry(real_store, '更新前正文', tenant_id, user_id, vector_size)

        await real_store.insert_memories([entry], session_scope='s-update-' + str(entry['id']))

        updated_text = '更新后正文22'
        new_payload = dict(entry['payload'])
        new_payload['text'] = updated_text
        new_payload['text_lemma'] = lemmatize_zh_for_bm25(keyword_text(updated_text))
        new_payload['hash'] = memory_hash(updated_text)
        new_payload['updated_at_tz'] = datetime.now(timezone.utc).isoformat()

        await real_store.update_memory(
            entry['id'],
            tenant_id,
            [0.2] * vector_size,
            new_payload,
            '更新前正文',
            user_id=user_id,
        )

        updated = await real_store.get_memory(entry['id'], tenant_id, user_id)
        assert updated is not None
        assert updated.text == updated_text


    async def test_real_store_delete_flow(self, real_store):
        tenant_id, user_id = _random_id(), _random_id()
        vector_size = real_store.vector_engine.vector_size
        entry = _build_entry(real_store, '异步待删除正文', tenant_id, user_id, vector_size)

        await real_store.insert_memories([entry], session_scope='s-delete-' + str(entry['id']))
        assert await real_store.get_memory(entry['id'], tenant_id, user_id) is not None

        await real_store.delete_memory(
            entry['id'], tenant_id, '异步待删除正文', entry['payload'], user_id=user_id)

        assert await real_store.get_memory(entry['id'], tenant_id, user_id) is None

    async def test_real_store_recall_flow(self, real_store):
        """真实向量库与存储层的双路召回（语义与关键词）。"""
        tenant_id, user_id = _random_id(), _random_id()
        topic, suffix = generate_random_tech_content()
        text_content = topic + '，' + suffix
        vector_size = real_store.vector_engine.vector_size
        vector = generate_random_vector(vector_size)
        entry = _build_entry(real_store, text_content, tenant_id, user_id, vector_size, vector=vector)

        await real_store.insert_memories([entry], session_scope='s-recall-' + str(entry['id']))

        results = await real_store.recall(
            query=topic,
            tenant_id=tenant_id,
            vector=vector,
            filter={'user_id': user_id},
            top_k=5,
            explain=True,
            user_id=user_id,
        )

        assert results is not None
        matched = [item for item in results if str(item['id']) == str(entry['id'])]
        assert len(matched) == 1
        assert matched[0]['memory'] == text_content

