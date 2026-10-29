'''stores.py 测试：MemoryStore 与 AsyncMemoryStore 的单元测试（Mock）
与正式数据库（PostgreSQL / Milvus）集成测试。

覆盖策略：
1. 单元测试：对 MemoryStore 和 AsyncMemoryStore 的核心方法进行全覆盖 Mock 测试，
   验证批量写入、异常回退、更新、删除、历史审计及双路召回的逻辑正确性。
2. 正式数据库集成测试：通过 --integration 参数隔离，在真实 PostgreSQL/pgvector
   和 Milvus 环境下验证真实的数据落库、更新、删除与检索行为。
3. 遵循项目规范：统一使用 pytest 与 pytest-asyncio 编写，妥善处理中文注释与异常捕获。
'''

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from loong_memories.nlp.lemmatization import lemmatize_zh_for_bm25
from loong_memories.storage.message.mysql_db import AsyncMySQLDB, MySQLDB
from loong_memories.storage.message.sqlite_db import AsyncSQLiteDB, SQLiteDB

try:
    import pytest_asyncio
    from conftest import CONFIG_DB_SETTINGS, CONFIG_VECTOR_SETTINGS
    from loong_memories.core.exceptions import StorageError
    from loong_memories.memory.stores import MemoryStore, AsyncMemoryStore, build_payload, keyword_text, memory_hash
    from loong_memories.storage.memories.base import VectorStoreResult
    from loong_memories.storage.memories.pg_store import PGVectorStore, AsyncPGVectorStore
    from loong_memories.storage.memories.milvus_store import MilvusVectorStore, AsyncMilvusVectorStore
    from loong_memories.storage.message.pg_db import PGDB, AsyncPGDB
    from loong_memories.storage.message.schema import HistoryRecord, Message
except Exception as e:
    pytest.skip(f'[SKIP INFO] File {__file__}：缺少依赖, 错误: {str(e)}',
                allow_module_level=True)

logger = logging.getLogger(__name__)

# ---------- 第一部分：单元测试（Mock） ----------

class TestMemoryStoreSync:
    """MemoryStore 同步存储封装单元测试"""

    @pytest.fixture
    def store_deps(self):
        vector = MagicMock()
        db = MagicMock()
        store = MemoryStore(vector=vector, db=db)
        return store, vector, db

    def test_insert_memories_success(self, store_deps):
        store, vector, db = store_deps
        entries = [
            {"id": 1, "vector": [0.1, 0.2], "payload": {"data": "test1", "user_id": "u1"}}
        ]
        messages = [Message(session_scope="s1", role="user", content="hello")]

        store.insert_memories(entries, session_scope="s1", messages=messages)

        vector.insert.assert_called_once()
        db.batch_add_history.assert_called_once()
        db.save_messages.assert_called_once()

    def test_insert_memories_fallback_on_batch_failure(self, store_deps):
        store, vector, db = store_deps
        vector.insert.side_effect = [Exception("Batch error"), None]

        entries = [
            {"id": 1, "vector": [0.1, 0.2], "payload": {"data": "test1", "user_id": "u1"}}
        ]

        store.insert_memories(entries, session_scope="s1")

        assert vector.insert.call_count == 2
        db.batch_add_history.assert_called_once()

    def test_insert_memories_all_failed_raises_error(self, store_deps):
        store, vector, db = store_deps
        vector.insert.side_effect = Exception("Persistent error")

        entries = [
            {"id": 1, "vector": [0.1, 0.2], "payload": {"data": "test1", "user_id": "u1"}}
        ]

        with pytest.raises(StorageError, match="记忆写入全部失败"):
            store.insert_memories(entries, session_scope="s1")

    def test_update_memory(self, store_deps):
        store, vector, db = store_deps
        vector.update.return_value = None

        store.update_memory(memory_id=1, vector=[0.3, 0.4], payload={"data": "new"}, previous="old")

        vector.update.assert_called_once_with(1, vector=[0.3, 0.4], payload={"data": "new"})
        db.batch_add_history.assert_called_once()

    def test_delete_memory(self, store_deps):
        store, vector, db = store_deps
        vector.delete.return_value = None

        store.delete_memory(memory_id=1, previous="old", payload={"data": "old"})

        vector.delete.assert_called_once_with(1)
        db.batch_add_history.assert_called_once()

    def test_get_memory_success(self, store_deps):
        store, vector, db = store_deps
        mock_res = VectorStoreResult(id=1, payload={"data": "content"})
        vector.get.return_value = mock_res

        res = store.get_memory(1)
        assert res == mock_res
        vector.get.assert_called_once_with(1)

    def test_load_history(self, store_deps):
        store, vector, db = store_deps
        db.get_last_messages.return_value = [Message(session_scope="s1", role="user", content="hi")]

        history = store.load_history("s1", limit=5)
        assert history == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
class TestAsyncMemoryStore:
    """AsyncMemoryStore 异步存储封装单元测试"""

    @pytest.fixture
    def async_store_deps(self):
        vector = AsyncMock()
        db = AsyncMock()
        store = AsyncMemoryStore(vector=vector, db=db)
        return store, vector, db

    async def test_async_insert_memories_success(self, async_store_deps):
        store, vector, db = async_store_deps
        entries = [
            {"id": 1, "vector": [0.1, 0.2], "payload": {"data": "test1", "user_id": "u1"}}
        ]
        messages = [Message(session_scope="s1", role="user", content="hello")]

        await store.insert_memories(entries, session_scope="s1", messages=messages)

        vector.insert.assert_awaited_once()
        db.batch_add_history.assert_awaited_once()

    async def test_async_update_memory(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.update.return_value = None

        await store.update_memory(memory_id=1, vector=[0.3, 0.4], payload={"data": "new"}, previous="old")

        vector.update.assert_awaited_once_with(1, vector=[0.3, 0.4], payload={"data": "new"})
        db.batch_add_history.assert_awaited_once()

    async def test_async_delete_memory(self, async_store_deps):
        store, vector, db = async_store_deps
        vector.delete.return_value = None

        await store.delete_memory(memory_id=1, previous="old", payload={"data": "old"})

        vector.delete.assert_awaited_once_with(1)
        db.batch_add_history.assert_awaited_once()


# ---------- 第二部分：正式数据库集成测试（Integration） ----------

pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    # 未传 --integration 时跳过下面的集成测试类
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）',
                allow_module_level=True)


def _random_id():
    return random.randint(1, 2 ** 31 - 1)


def generate_random_tech_content() -> tuple[str, str]:
    """生成随机的技术领域测试文本"""
    # 核心技术主题库
    topics = [
        "人工智能和机器学习是现代软件工程的核心技术",
        "基于领域驱动设计（DDD）的大型分布式微服务架构演进",
        "利用 pgvector 和 PostgreSQL 构建高性能的检索增强生成（RAG）系统",
        "高并发场景下的数据库事务传播与连接池优化策略",
        "基于大语言模型的长期记忆机制与上下文压缩算法研究",
        "云原生环境中的容器化部署与 CI/CD 自动化流水线管理",
        "多租户架构下的数据隔离方案与安全风险控制"
    ]

    # 扩展描述/业务结论库
    suffixes = [
        "能够有效降低系统耦合度并大幅提升整体吞吐量。",
        "在处理海量并发请求与复杂事务时表现出优异的稳定性。",
        "是当前企业级研发团队在数字化转型中面临的重要技术挑战。",
        "通过精细化指标评估与底层索引调优实现了显著的性能跃升。"
    ]

    # 随机组合
    return random.choice(topics),random.choice(suffixes)

def generate_random_vector(dim: int = 1024) -> list[float]:
    """生成指定维度的向量，数值在 0.15 附近随机波动"""
    return [round(0.15 + random.uniform(-0.02, 0.02), 4) for _ in range(dim)]

DB_BACKENDS = ["sqlite", "mysql", "pg"] # ["sqlite", "mysql", "pg"]
VECTOR_BACKENDS = ["milvus", "pg"]  # ["milvus", "pg"]

@pytest.fixture(scope="module", params=DB_BACKENDS)
def db_backend(request):
    return request.param

@pytest.fixture(scope="module", params=VECTOR_BACKENDS)
def vector_backend(request):
    return request.param

class TestMemoryStoreIntegration:
    """MemoryStore 结合真实 PostgreSQL 与 PGVectorStore 的端到端集成测试"""

    table_prefix = "store_test_"
    collection_name = "store_test_sync_vector"

    @pytest.fixture(scope='module')
    def real_store(self, db_backend, vector_backend):
        if '--integration' not in sys.argv:
            pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）',
                        allow_module_level=True)

        logger.info(f"db_backend:{db_backend} vector_backend:{vector_backend}")

        # 初始化真实关系数据库引擎
        if "pg" == db_backend:
            db_conf = CONFIG_DB_SETTINGS.pg
            db_class = PGDB

        elif "mysql" == db_backend:
            db_conf = CONFIG_DB_SETTINGS.mysql
            db_class = MySQLDB

        else:
            db_conf = CONFIG_DB_SETTINGS.sqlite
            db_class = SQLiteDB

        new_db_conf = {
            **db_conf.model_dump(),  # 核心：将 PGSettings 转换为 Python 字典
            "table_prefix": TestMemoryStoreIntegration.table_prefix,
        }

        logger.info(f"db:{str(db_class)} config:{new_db_conf}")

        db = db_class(**new_db_conf)

        # 初始化真实 PG 向量库引擎
        if "milvus" == vector_backend:
            vec_conf = CONFIG_VECTOR_SETTINGS.milvus
            vector_class = MilvusVectorStore
        else:
            vec_conf = CONFIG_VECTOR_SETTINGS.pg
            vector_class = PGVectorStore

        new_vector_conf = {
            **vec_conf.model_dump(),
            "collection_name":TestMemoryStoreIntegration.collection_name
        }
        logger.info(f"vector_db:{str(vector_class)} config:{new_vector_conf}")

        vector = vector_class(**new_vector_conf)

        store = MemoryStore(vector=vector, db=db)

        try:
            store.start()
        except Exception as exc:
            pytest.skip(f'启动 Store 失败:' + str(exc))

        yield store

        # 清理资源
        if hasattr(vector, 'engine') and vector.engine:
            vector.engine.dispose()
        if hasattr(db, 'engine') and db.engine:
            db.engine.dispose()

    def test_real_store_insert_and_get(self, real_store):
        memory_id = _random_id()
        scope = "session-" + str(memory_id)
        payload = build_payload("测试真实落库正文", {"user_id": "123", "tenant_id": 1})
        entry = {
            "id": memory_id,
            "vector": [0.1] * real_store.vector.vector_size,
            "payload": payload
        }
        messages = [Message(session_scope=scope, role="user", content="测试真实落库正文")]

        # 执行批量写入（包含向量、历史、消息）
        real_store.insert_memories([entry], session_scope=scope, messages=messages)

        # 验证向量库能正常读到
        stored_item = real_store.get_memory(memory_id)
        assert stored_item is not None
        assert stored_item.id == memory_id

        # 验证历史记录已加载
        history = real_store.load_history(scope, limit=5)
        assert len(history) == 1
        assert history[0]["content"] == "测试真实落库正文"

    def test_real_store_update_flow(self, real_store):
        memory_id = _random_id()
        payload = build_payload("更新前正文", {"user_id": "123", "tenant_id": 1})
        entry = {
            "id": memory_id,
            "vector": [0.2] * real_store.vector.vector_size,
            "payload": payload
        }
        real_store.insert_memories([entry], session_scope="s-update")

        update_text = "更新后正文22"
        query_lemmatized = lemmatize_zh_for_bm25(keyword_text(update_text))
        # 直接构造更新后的 payload 传给 store.update_memory
        new_payload = dict(payload)
        new_payload["text"] = update_text
        new_payload["updated_at_tz"] = datetime.now(timezone.utc).isoformat()
        new_payload["text_lemma"] = query_lemmatized
        new_payload["hash"] = memory_hash(update_text)

        real_store.update_memory(
            memory_id=memory_id,
            vector=[0.3] * real_store.vector.vector_size,
            payload=new_payload,
            previous="更新前正文"
        )

        updated_item = real_store.get_memory(memory_id)
        assert updated_item.text == "更新后正文22"

    def test_real_store_delete_flow(self, real_store):
        memory_id = _random_id()
        payload = build_payload("待删除正文", {"user_id": "123", "tenant_id": 1})
        entry = {
            "id": memory_id,
            "vector": [0.4] * real_store.vector.vector_size,
            "payload": payload
        }
        real_store.insert_memories([entry], session_scope="s-delete")

        assert real_store.get_memory(memory_id) is not None

        # 执行删除
        real_store.delete_memory(memory_id, payload.get("text"), payload)
        assert real_store.get_memory(memory_id) is None

    def test_real_store_recall_flow(self, real_store):
        """测试真实 PG 向量库与存储层的双路召回（语义与关键词检索）"""
        memory_id = _random_id()
        session_scope = "s-recall-" + str(memory_id)

        topic,suffix = generate_random_tech_content()
        vectors = generate_random_vector()

        # 准备含有中文分词/关键词的正文
        text_content = f"{topic}，{suffix}"
        payload = build_payload(text_content, {"user_id": 456, "tenant_id": 1})

        entry = {
            "id": memory_id,
            "vector": vectors,
            "payload": payload
        }

        # 1. 写入测试数据
        real_store.insert_memories([entry], session_scope=session_scope)

        # 2. 调用 recall 方法进行双路召回
        # 传入 query 文本、模拟的向量、过滤条件以及 top_k
        results = real_store.recall(
            query=topic,
            vector=vectors,
            filters={"user_id": 456},
            top_k=5,
            explain=True
        )

        # 3. 断言召回结果
        assert results is not None
        assert len(results) >= 1
        # 验证能准确召回刚才写入的那条记忆
        matched = [r for r in results if str(r["id"]) == str(memory_id)]
        assert len(matched) == 1
        assert matched[0]["memory"] == text_content





DB_ASYNC_BACKENDS = ["sqlite_async", "mysql_async", "pg_async"] # ["sqlite_async", "mysql_async", "pg_async"]
VECTOR_ASYNC_BACKENDS = ["milvus_async", "pg_async"]  # ["milvus_async", "pg_async"]

@pytest_asyncio.fixture(scope="module", loop_scope='module', params=DB_ASYNC_BACKENDS)
def db_async_backend(request):
    return request.param

@pytest_asyncio.fixture(scope="module", loop_scope='module', params=VECTOR_ASYNC_BACKENDS)
def vector_async_backend(request):
    return request.param

@pytest.mark.asyncio(loop_scope='module')
class TestAsyncMemoryStoreIntegration:
    """AsyncMemoryStore 结合真实 PostgreSQL 与 PGVectorStore 的端到端集成测试"""

    table_prefix = "store_test_async_"
    collection_name = "store_test_async_vector"

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def real_store(self, db_async_backend, vector_async_backend):
        if '--integration' not in sys.argv:
            pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）',
                        allow_module_level=True)

        logger.info(f"db_async_backend:{db_async_backend} vector_async_backend:{vector_async_backend}")

        # 初始化真实关系数据库引擎
        if "pg_async" == db_async_backend:
            db_conf = CONFIG_DB_SETTINGS.pg
            db_class = AsyncPGDB
        elif "mysql_async" == db_async_backend:
            db_conf = CONFIG_DB_SETTINGS.mysql
            db_class = AsyncMySQLDB
        else:
            db_conf = CONFIG_DB_SETTINGS.sqlite
            db_class = AsyncSQLiteDB

        new_db_conf = {
            **db_conf.model_dump(),  # 核心：将 PGSettings 转换为 Python 字典
            "table_prefix": TestAsyncMemoryStoreIntegration.table_prefix,
        }

        logger.info(f"db_async:{str(db_class)} config:{new_db_conf}")

        db = db_class(**new_db_conf)


        # 初始化真实 PG 向量库引擎
        if "milvus_async" == vector_async_backend:
            vec_conf = CONFIG_VECTOR_SETTINGS.milvus
            vector_class = AsyncMilvusVectorStore
        else:
            vec_conf = CONFIG_VECTOR_SETTINGS.pg
            vector_class = AsyncPGVectorStore

        new_vector_conf = {
            **vec_conf.model_dump(),
            "collection_name":TestAsyncMemoryStoreIntegration.collection_name
        }
        logger.info(f"vector_db:{str(vector_class)} config:{new_vector_conf}")

        vector = vector_class(**new_vector_conf)


        store = AsyncMemoryStore(vector=vector, db=db)

        try:
            await store.start()
        except Exception as exc:
            pytest.skip(f'启动 Store 失败:' + str(exc))

        yield store

        # 清理资源
        if hasattr(vector, 'engine') and vector.engine:
            await vector.engine.dispose()
        if hasattr(db, 'engine') and db.engine:
            await db.engine.dispose()

    async def test_real_store_insert_and_get(self, real_store):
        memory_id = _random_id()
        scope = "session-" + str(memory_id)
        payload = build_payload("测试真实落库正文", {"user_id": "123", "tenant_id": 1})
        entry = {
            "id": memory_id,
            "vector": [0.1] * real_store.vector.vector_size,
            "payload": payload
        }
        messages = [Message(session_scope=scope, role="user", content="测试真实落库正文")]

        # 执行批量写入（包含向量、历史、消息）
        await real_store.insert_memories([entry], session_scope=scope, messages=messages)

        # 验证向量库能正常读到
        stored_item = await real_store.get_memory(memory_id)
        assert stored_item is not None
        assert stored_item.id == memory_id

        # 验证历史记录已加载
        history = await real_store.load_history(scope, limit=5)
        assert len(history) == 1
        assert history[0]["content"] == "测试真实落库正文"

    async def test_real_store_update_flow(self, real_store):
        memory_id = _random_id()
        payload = build_payload("更新前正文", {"user_id": "123", "tenant_id": 1})
        entry = {
            "id": memory_id,
            "vector": [0.2] * real_store.vector.vector_size,
            "payload": payload
        }
        await real_store.insert_memories([entry], session_scope="s-update")

        update_text = "更新后正文22"
        query_lemmatized = lemmatize_zh_for_bm25(keyword_text(update_text))
        # 直接构造更新后的 payload 传给 store.update_memory
        new_payload = dict(payload)
        new_payload["text"] = update_text
        new_payload["updated_at_tz"] = datetime.now(timezone.utc).isoformat()
        new_payload["text_lemma"] = query_lemmatized
        new_payload["hash"] = memory_hash(update_text)

        await real_store.update_memory(
            memory_id=memory_id,
            vector=[0.3] * real_store.vector.vector_size,
            payload=new_payload,
            previous="更新前正文"
        )

        updated_item = await real_store.get_memory(memory_id)
        assert updated_item.text == "更新后正文22"

    async def test_real_store_delete_flow(self, real_store):
        memory_id = _random_id()
        payload = build_payload("待删除正文", {"user_id": "123", "tenant_id": 1})
        entry = {
            "id": memory_id,
            "vector": [0.4] * real_store.vector.vector_size,
            "payload": payload
        }
        await real_store.insert_memories([entry], session_scope="s-delete")

        assert await real_store.get_memory(memory_id) is not None

        # 执行删除
        await real_store.delete_memory(memory_id, payload.get("text"), payload)
        assert await real_store.get_memory(memory_id) is None

    async def test_real_store_recall_flow(self, real_store):
        """测试真实 PG 向量库与存储层的双路召回（语义与关键词检索）"""
        memory_id = _random_id()
        session_scope = "s-recall-" + str(memory_id)

        # 准备含有中文分词/关键词的正文
        topic, suffix = generate_random_tech_content()
        vectors = generate_random_vector()

        text_content = f"{topic}，{suffix}"
        payload = build_payload(text_content, {"user_id": 456, "tenant_id": 1})

        entry = {
            "id": memory_id,
            "vector": vectors,
            "payload": payload
        }

        # 1. 写入测试数据
        await real_store.insert_memories([entry], session_scope=session_scope)

        # 2. 调用 recall 方法进行双路召回
        # 传入 query 文本、模拟的向量、过滤条件以及 top_k
        results = await real_store.recall(
            query=topic,
            vector=vectors,
            filters={"user_id": 456},
            top_k=5,
            explain=True
        )

        # 3. 断言召回结果
        assert results is not None
        assert len(results) >= 1
        # 验证能准确召回刚才写入的那条记忆
        matched = [r for r in results if str(r["id"]) == str(memory_id)]
        assert len(matched) == 1
        assert matched[0]["memory"] == text_content

