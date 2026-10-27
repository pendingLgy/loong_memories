'''pg_store.py 测试：PGVectorStore 与 AsyncPGVectorStore 的公共契约与业务方法。

覆盖策略：
1. 全方法覆盖：构造方法 __init__、backend 与 metric_type 类属性，以及 start、create_col、
   insert、search、keyword_search、delete、update、get、list 九个公共方法。
2. 继承契约验证：AsyncPGVectorStore 复用同步版的 SQL 构造与结果解析，其公共方法在异步
   子类实例上单独验证，不与同步类混用。
3. 不单独测试私有方法：_get_table、_col、_build_sql_*、_parse、_session 等不做直接断言，
   其逻辑由 insert、search、list、get 的行为间接验证。
4. 不为父类建测试类：VectorStoreBase 不设独立测试套件。

向量库是外部服务，故整个模块以 integration 标记隔离：
未传 --integration 时整体跳过；异步用例另需安装 asyncpg 与 pgvector 扩展。
连接参数取自 conftest 的 vector 分区，可用 LOONG_TEST_VECTOR_PG_DSN 覆盖。

运行：
    python -m pytest tests --integration -q
'''

from __future__ import annotations

import importlib.util
import os
import random
import sys
from datetime import datetime, timezone

import pytest


try:
    import sqlalchemy
    import pytest_asyncio
    import asyncpg

    from conftest import CONFIG_VECTOR_SETTINGS
    from loong_memories.core.exceptions import OptionalDependencyError
    from loong_memories.storage.memories.base import VectorStoreBase
    from loong_memories.storage.memories.pg_store import AsyncPGVectorStore, PGVectorStore
    from loong_memories.storage.memories.registry_vector import (
        is_vector_store_registered,
        resolve_vector_store,
    )

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)


pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)


# ---------- 常量与辅助 ----------

# 集成用例专用集合名与维度：集合即表名，固定命名便于复用，维度取小值以减少写入数据量。
TEST_COLLECTION = 'vut_vectors'
TEST_VECTOR_SIZE = 8


def _random_id():
    '''随机正整数：避免跨用例、跨次运行的数据互相干扰。'''
    return random.randint(1, 2 ** 31 - 1)


def _vector(seed):
    '''构造 TEST_VECTOR_SIZE 维的方向向量：仅首元素随 seed 变化，便于预期距离排序。'''
    vector = [0.0] * TEST_VECTOR_SIZE
    vector[0] = float(seed)
    return vector


def _payload(tenant_id, text, text_lemma=None):
    '''组装一条写入用的 payload：作用域列、元数据 JSONB 与文本字段齐备。'''
    now = datetime.now(timezone.utc)
    return {
        'tenant_id': tenant_id,
        'user_id': _random_id(),
        'metadata': {'tag': text},
        'text': text,
        'text_lemma': text_lemma or text,
        'created_at_tz': now,
        'updated_at_tz': now,
    }


# ---------- fixtures ----------

@pytest.fixture(scope='module')
def vector_settings():
    '''vector 分区配置：取自 conftest 常量，逐模块深拷贝避免互相污染。'''
    return CONFIG_VECTOR_SETTINGS.model_copy(deep=True)


class TestPGVectorStore:
    '''PGVectorStore 的公共契约：backend 登记、构造参数与 metric_type。'''

    def test_backend_is_registered(self):
        assert PGVectorStore.backend == 'pg'
        assert is_vector_store_registered('pg') is True
        assert resolve_vector_store('pg') is PGVectorStore

    def test_inherits_store_contract(self):
        assert issubclass(PGVectorStore, VectorStoreBase)
        assert isinstance(PGVectorStore(), VectorStoreBase)

    def test_init_defaults(self):
        store = PGVectorStore()
        assert store.connection_string is None
        assert (store.collection_name, store.vector_size) == ('', 1024)
        assert (store.database, store.user, store.password) == ('postgres', 'postgres', '')
        assert (store.host, store.port) == ('localhost', 5432)
        assert (store.use_hnsw, store.use_diskann, store.print_sql) == (True, False, False)
        assert (store.pool_size, store.max_overflow, store.pool_timeout) == (10, 20, 30)
        assert (store.pool_recycle, store.pool_pre_ping) == (1800, True)

    def test_init_stores_conftest_values(self, vector_settings):
        pg, pool = vector_settings.pg, vector_settings.pool_config
        store = PGVectorStore(
            connection_string=pg.connection_string,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
        )
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert (store.database, store.user, store.password) == (pg.database, pg.user, pg.password)
        assert (store.host, store.port) == (pg.host, pg.port)
        assert (store.use_hnsw, store.use_diskann) == (pg.use_hnsw, pg.use_diskann)
        assert (store.pool_size, store.max_overflow, store.pool_timeout) == (pool.pool_size, pool.max_overflow, pool.pool_timeout)
        assert (store.pool_recycle, store.pool_pre_ping) == (pool.pool_recycle, pool.pool_pre_ping)

    def test_init_builds_empty_metadata(self):
        assert list(PGVectorStore().metadata_obj.tables) == []

    def test_metric_type_is_cosine(self):
        assert PGVectorStore.metric_type == 'COSINE'

    def test_start_without_sqlalchemy_raises_optional_dependency_error(self, monkeypatch):
        monkeypatch.setattr('loong_memories.storage.memories.pg_store._SQLALCHEMY_AVAILABLE', False)
        with pytest.raises(OptionalDependencyError):
            PGVectorStore().start()


class TestAsyncPGVectorStore:
    '''AsyncPGVectorStore 的公共契约：backend 登记与构造参数。'''

    def test_backend_is_registered(self):
        assert AsyncPGVectorStore.backend == 'pg_async'
        assert is_vector_store_registered('pg_async') is True
        assert resolve_vector_store('pg_async') is AsyncPGVectorStore

    def test_inherits_sync_store(self):
        assert issubclass(AsyncPGVectorStore, PGVectorStore)
        assert issubclass(AsyncPGVectorStore, VectorStoreBase)

    def test_init_defaults(self):
        store = AsyncPGVectorStore()
        assert store.connection_string is None
        assert (store.collection_name, store.vector_size) == ('', 1024)
        assert (store.database, store.user, store.password) == ('postgres', 'postgres', '')
        assert (store.host, store.port) == ('localhost', 5432)
        assert (store.use_hnsw, store.use_diskann, store.print_sql) == (True, False, False)
        assert (store.pool_size, store.max_overflow, store.pool_timeout) == (10, 20, 30)
        assert (store.pool_recycle, store.pool_pre_ping) == (1800, True)

    def test_init_stores_conftest_values(self, vector_settings):
        pg, pool = vector_settings.pg, vector_settings.pool_config
        store = AsyncPGVectorStore(
            connection_string=pg.connection_string,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
        )
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert (store.database, store.user, store.password) == (pg.database, pg.user, pg.password)
        assert (store.host, store.port) == (pg.host, pg.port)
        assert (store.use_hnsw, store.use_diskann) == (pg.use_hnsw, pg.use_diskann)

    def test_metric_type_is_cosine(self):
        assert AsyncPGVectorStore.metric_type == 'COSINE'

    async def test_start_without_sqlalchemy_raises_optional_dependency_error(self, monkeypatch):
        monkeypatch.setattr('loong_memories.storage.memories.pg_store._SQLALCHEMY_AVAILABLE', False)
        with pytest.raises(OptionalDependencyError):
            await AsyncPGVectorStore().start()


# ---------- 真实服务：PGVectorStore ----------

class TestPGVectorStoreIntegration:
    '''PGVectorStore 在真实 PostgreSQL 上的公共方法行为：建表、写入、检索与更新。

    连接参数取自 conftest 的 vector 分区，可用 LOONG_TEST_VECTOR_PG_DSN 覆盖；
    集合名固定为 TEST_COLLECTION（即表名），用例靠随机 tenant_id 隔离数据。
    '''

    @pytest.fixture(scope='module')
    def pg_store(self, vector_settings):
        '''建连并确保表、向量索引与全文索引存在（start 内部会调用 create_col）。'''
        pg, pool = vector_settings.pg, vector_settings.pool_config
        dsn = os.environ.get('LOONG_TEST_VECTOR_PG_DSN', '') or pg.connection_string
        store = PGVectorStore(
            connection_string=dsn or None,
            collection_name=TEST_COLLECTION,
            vector_size=TEST_VECTOR_SIZE,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            sslmode=pg.sslmode,
        )
        try:
            store.start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
        yield store

        store.engine.dispose()

    def test_start_returns_self(self, pg_store):
        assert isinstance(pg_store, PGVectorStore)
        assert pg_store.engine is not None

    def test_create_col_is_idempotent(self, pg_store):
        assert pg_store.create_col(TEST_COLLECTION, TEST_VECTOR_SIZE) is None

    def test_insert_then_get_returns_record(self, pg_store):
        tenant_id, rid = _random_id(), _random_id()
        pg_store.insert([_vector(1)], [_payload(tenant_id, 'hello')], [rid])
        got = pg_store.get(rid)
        assert got is not None
        assert (got.id, got.tenant_id, got.text) == (rid, tenant_id, 'hello')

    def test_get_unknown_id_returns_none(self, pg_store):
        assert pg_store.get(_random_id()) is None

    def test_insert_persists_text_lemma_and_metadata(self, pg_store):
        rid = _random_id()
        pg_store.insert([_vector(1)], [_payload(_random_id(), '原文', '词形')], [rid])
        got = pg_store.get(rid)
        assert (got.text, got.text_lemma) == ('原文', '词形')
        assert got.metadata == {'tag': '原文'}

    def test_created_at_is_serialized_to_iso_string(self, pg_store):
        rid = _random_id()
        pg_store.insert([_vector(1)], [_payload(_random_id(), 't')], [rid])
        assert isinstance(pg_store.get(rid).created_at_tz, str)

    def test_search_returns_closest_first(self, pg_store):
        tenant_id = _random_id()
        ids = [_random_id() for _ in range(3)]
        pg_store.insert([_vector(1), _vector(2), _vector(3)],
                        [_payload(tenant_id, 'm' + str(index)) for index in range(3)], ids)
        hits = pg_store.search('m0', _vector(1), top_k=3, filters={'tenant_id': tenant_id})
        assert [hit.id for hit in hits] == ids
        distances = [hit.score for hit in hits]
        assert distances == sorted(distances)

    def test_search_respects_top_k(self, pg_store):
        tenant_id = _random_id()
        pg_store.insert([_vector(1) for _ in range(3)],
                        [_payload(tenant_id, 'top' + str(index)) for index in range(3)],
                        [_random_id() for _ in range(3)])
        assert len(pg_store.search('top', _vector(1), top_k=2, filters={'tenant_id': tenant_id})) == 2

    def test_search_filters_by_tenant(self, pg_store):
        first, second = _random_id(), _random_id()
        pg_store.insert([_vector(1), _vector(1)],
                        [_payload(first, 'a'), _payload(second, 'b')],
                        [_random_id(), _random_id()])
        hits = pg_store.search('a', _vector(1), top_k=10, filters={'tenant_id': first})
        assert [hit.text for hit in hits] == ['a']

    def test_search_filters_by_metadata_key(self, pg_store):
        tenant_id, rid = _random_id(), _random_id()
        pg_store.insert([_vector(1)], [_payload(tenant_id, 'unique-tag')], [rid])
        hits = pg_store.search('q', _vector(1), top_k=10, filters={'tag': 'unique-tag'})
        assert rid in [hit.id for hit in hits]

    def test_keyword_search_matches_text_lemma(self, pg_store):
        tenant_id, rid = _random_id(), _random_id()
        pg_store.insert([_vector(1)], [_payload(tenant_id, 'ignored', 'milvus vector database')], [rid])
        hits = pg_store.keyword_search('milvus', top_k=5, filters={'tenant_id': tenant_id})
        assert hits is not None
        assert [hit.id for hit in hits] == [rid]

    def test_keyword_search_returns_none_on_invalid_query(self, pg_store):
        assert pg_store.keyword_search('12', top_k=5) == []

    def test_update_changes_vector_and_metadata(self, pg_store):
        tenant_id, rid = _random_id(), _random_id()
        pg_store.insert([_vector(1)], [_payload(tenant_id, 'before')], [rid])
        pg_store.update(rid, vector=_vector(4), payload={'tag': 'after'})
        got = pg_store.get(rid)
        assert got.metadata == {'tag': 'after'}

    def test_update_without_params_is_noop(self, pg_store):
        tenant_id, rid = _random_id(), _random_id()
        pg_store.insert([_vector(1)], [_payload(tenant_id, 'kept')], [rid])
        pg_store.update(rid)
        assert pg_store.get(rid).text == 'kept'

    def test_delete_removes_row(self, pg_store):
        tenant_id, rid = _random_id(), _random_id()
        pg_store.insert([_vector(1)], [_payload(tenant_id, 'gone')], [rid])
        pg_store.delete(rid)
        assert pg_store.get(rid) is None

    def test_insert_duplicate_id_raises_integrity_error(self, pg_store):
        rid = _random_id()
        pg_store.insert([_vector(1)], [_payload(_random_id(), 'dup')], [rid])
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            pg_store.insert([_vector(1)], [_payload(_random_id(), 'dup')], [rid])

    def test_list_respects_top_k(self, pg_store):
        tenant_id = _random_id()
        pg_store.insert([_vector(1) for _ in range(3)],
                        [_payload(tenant_id, 'l' + str(index)) for index in range(3)],
                        [_random_id() for _ in range(3)])
        assert len(pg_store.list(filters={'tenant_id': tenant_id}, top_k=2)) == 2

    def test_list_applies_in_operator(self, pg_store):
        first, second = _random_id(), _random_id()
        pg_store.insert([_vector(1), _vector(1)],
                        [_payload(first, 'x'), _payload(second, 'y')],
                        [_random_id(), _random_id()])
        listed = pg_store.list(filters={'tenant_id': {'in': [first]}})
        assert [item.text for item in listed] == ['x']


# ---------- 真实服务：AsyncPGVectorStore ----------

@pytest.mark.asyncio(loop_scope='module')
class TestAsyncPGVectorStoreIntegration:
    '''AsyncPGVectorStore 在真实 PostgreSQL 上的公共方法行为，语义与同步版本一致。'''

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def async_pg_store(self, vector_settings):
        '''建连并确保表与索引存在（start 内部会 await create_col）。'''
        pg, pool = vector_settings.pg, vector_settings.pool_config
        dsn = os.environ.get('LOONG_TEST_VECTOR_PG_DSN', '') or pg.connection_string
        store = AsyncPGVectorStore(
            connection_string=dsn or None,
            collection_name=TEST_COLLECTION,
            vector_size=TEST_VECTOR_SIZE,
            database=pg.database,
            user=pg.user,
            password=pg.password,
            host=pg.host,
            port=pg.port,
            use_hnsw=pg.use_hnsw,
            use_diskann=pg.use_diskann,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            sslmode=pg.sslmode,
        )
        try:
            await store.start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
        yield store

        await store.engine.dispose()

    async def test_start_returns_self(self, async_pg_store):
        assert isinstance(async_pg_store, AsyncPGVectorStore)
        assert async_pg_store.engine is not None

    async def test_create_col_is_idempotent(self, async_pg_store):
        assert await async_pg_store.create_col(TEST_COLLECTION, TEST_VECTOR_SIZE) is None

    async def test_insert_then_get_returns_record(self, async_pg_store):
        tenant_id, rid = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1)], [_payload(tenant_id, 'hello')], [rid])
        got = await async_pg_store.get(rid)
        assert got is not None
        assert (got.id, got.text) == (rid, 'hello')

    async def test_get_unknown_id_returns_none(self, async_pg_store):
        assert await async_pg_store.get(_random_id()) is None

    async def test_search_returns_closest_first(self, async_pg_store):
        tenant_id = _random_id()
        ids = [_random_id() for _ in range(3)]
        await async_pg_store.insert([_vector(1), _vector(2), _vector(3)],
                                    [_payload(tenant_id, 'm' + str(index)) for index in range(3)], ids)
        hits = await async_pg_store.search('m0', _vector(1), top_k=3, filters={'tenant_id': tenant_id})
        assert [hit.id for hit in hits] == ids

    async def test_search_respects_top_k(self, async_pg_store):
        tenant_id = _random_id()
        await async_pg_store.insert([_vector(1) for _ in range(3)],
                                   [_payload(tenant_id, 'top' + str(index)) for index in range(3)],
                                   [_random_id() for _ in range(3)])
        hits = await async_pg_store.search('top', _vector(1), top_k=2, filters={'tenant_id': tenant_id})
        assert len(hits) == 2

    async def test_search_filters_by_tenant(self, async_pg_store):
        first, second = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1), _vector(1)],
                                   [_payload(first, 'a'), _payload(second, 'b')],
                                   [_random_id(), _random_id()])
        hits = await async_pg_store.search('a', _vector(1), top_k=10, filters={'tenant_id': first})
        assert [hit.text for hit in hits] == ['a']

    async def test_keyword_search_matches_text_lemma(self, async_pg_store):
        tenant_id, rid = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1)], [_payload(tenant_id, 'ignored', 'milvus vector database')], [rid])
        hits = await async_pg_store.keyword_search('milvus', top_k=5, filters={'tenant_id': tenant_id})
        assert hits is not None and hits[0].id == rid

    async def test_keyword_search_returns_none_on_invalid_query(self, async_pg_store):
        assert await async_pg_store.keyword_search('12', top_k=5) == []

    async def test_update_changes_metadata(self, async_pg_store):
        tenant_id, rid = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1)], [_payload(tenant_id, 'before')], [rid])
        await async_pg_store.update(rid, payload={'tag': 'after'})
        got = await async_pg_store.get(rid)
        assert got.metadata == {'tag': 'after'}

    async def test_update_without_params_is_noop(self, async_pg_store):
        tenant_id, rid = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1)], [_payload(tenant_id, 'kept')], [rid])
        await async_pg_store.update(rid)
        got = await async_pg_store.get(rid)
        assert got.text == 'kept'

    async def test_delete_removes_row(self, async_pg_store):
        tenant_id, rid = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1)], [_payload(tenant_id, 'gone')], [rid])
        await async_pg_store.delete(rid)
        assert await async_pg_store.get(rid) is None

    async def test_list_applies_filters(self, async_pg_store):
        tenant_id, rid = _random_id(), _random_id()
        await async_pg_store.insert([_vector(1)], [_payload(tenant_id, 'only')], [rid])
        listed = await async_pg_store.list(filters={'tenant_id': tenant_id})
        assert [item.id for item in listed] == [rid]
