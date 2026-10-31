'''vector_engine.py 测试：VectorEngine 与 AsyncVectorEngine 的公共契约。

覆盖策略：
1. 全方法覆盖：构造参数、backend、collection_name、vector_size、start、
   insert、update、delete、search、keyword_search、get、get_by_filter、list、
   __enter__、__exit__，以及模块级 build_vector_store、create_vector_engine、
   build_vector_engine。
2. 真实服务：同步与异步引擎分别接入真实 PostgreSQL 向量库，集合固定为
   TEST_COLLECTION（表名），主键与 tenant_id、user_id 全部来自 _next_id。
3. 引擎层 tenant_id 与 user_id 一律透传给底层 store；insert 的租户实际取自
   payload，故用例保证 payload 内 tenant_id 与调用实参一致。
4. search、keyword_search、list 不传 user_id 形参：父类会把租户值写进
   user_id 维度，用户维度统一用 filters 里的 user_id 键表达。
5. 向量与文本全随机：seed 取系统熵，文本从 _TEXTS 抽样，断言只比对写入值。
6. 不为 VectorStoreBase、PGVectorStore、AsyncPGVectorStore 建套件，也不直接
   断言私有方法，其逻辑经本文件的公共 API 间接验证。

运行：
    python -m pytest tests --integration -q -k vector_engine
'''

from __future__ import annotations

import itertools
import os
import random
import sys

import pytest

try:
    import asyncpg
    import pytest_asyncio

    from conftest import CONFIG_VECTOR_SETTINGS
    from loong_memories.core.exceptions import ConfigError, StorageError
    from loong_memories.storage.memories.pg_store import (
        AsyncPGVectorStore,
        PGVectorStore,
    )
    from loong_memories.storage.memories.vector_engine import (
        AsyncVectorEngine,
        VectorEngine,
        build_vector_engine,
        build_vector_store,
        create_vector_engine,
    )

except ImportError as _exc:
    pytest.skip('[SKIP INFO] File ' + __file__ + '：缺少 ' + str(_exc), allow_module_level=True)


pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip('[SKIP INFO] integration 测试，默认跳过（请使用 --integration）', allow_module_level=True)


# ---------- 常量与辅助 ----------

# PG 的集合即表，且 create_col 内 GIN 索引硬编码写在 vut_vectors 上，
# 故集合名必须保持该值；维度取小值以减少写入数据量。
TEST_COLLECTION = 'vut_vectors'
TEST_VECTOR_SIZE = 8

# 进程内唯一正整数发生器：基数取系统熵，之后严格自增。
_ID_BASE = random.SystemRandom().randrange(1, 2 ** 40)
_ID_SEQ = itertools.count(1)

# 系统熵随机源：向量 seed 也走系统熵，用例里不出现固定数字 seed。
_TRNG = random.SystemRandom()


def _next_id():
    '''返回本次运行内唯一的正整数。'''
    return _ID_BASE + next(_ID_SEQ)


def _random_seed():
    '''返回一次性随机 seed：每次调用都从系统熵取值。'''
    return _TRNG.randrange(1, 2 ** 62)


def _vector(seed):
    '''构造 TEST_VECTOR_SIZE 维完全随机向量：同一 seed 逐位相同。'''
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(TEST_VECTOR_SIZE)]


# 文本池：跨领域、跨句式的异构文本；keyword_search 用例只用其中含英文单词
# 的条目，因为 PG 的 simple 分词不切分中文。
_TEXTS = [
    'vector database stores embeddings for similarity search',
    'retrieval augmented generation combines search with language models',
    'cosine distance measures the angle between two vectors',
    'postgresql extension adds nearest neighbour search',
    'the engine wraps a vector store and forwards tenant scope',
    'keyword search relies on full text ranking functions',
    'batch insert validates payload count and identifier count',
    'memory handler persists conversation turns into the store',
    'multi tenant isolation requires tenant filters on every query',
    'snowflake identifiers keep primary keys globally unique',
    'a rolling summary compresses long dialogue history',
    'embedding dimension mismatch raises a storage error',
    'list returns rows limited by the requested top k value',
    'update refreshes payload while keeping the stored vector',
    'delete removes a single record scoped by its tenant',
    'the default top k value is used when the argument is None',
    '异步引擎在真实数据库上复用同一套查询构造',
    '向量检索与关键词检索的结果需要合并打分',
    '租户维度必须参与每一次读写的定位条件',
    '长期记忆需要保留最近若干轮完整对话',
]


def _random_text():
    '''从文本池随机取一条。'''
    return _TRNG.choice(_TEXTS)


def _distinct_texts(count):
    '''无放回抽取 count 条互不相同的文本。'''
    return random.sample(_TEXTS, count)


def _payload(tenant_id, user_id, text=None, text_lemma=None, **extra):
    '''组装一条写入 payload；user_id 缺省时自行分配，因 vut_vectors 的 user_id 列为 NOT NULL。'''
    user_id = _next_id() if user_id is None else user_id
    content = _random_text() if text is None else text
    payload = dict(
        tenant_id=tenant_id,
        user_id=user_id,
        tag=content,
        text=content,
        text_lemma=text_lemma or content,
    )
    payload.update(extra)
    return payload


def _scope(tenant_id, **extra):
    '''构造非空 filter 字典：显式带上 tenant_id 保证租户条件生效。'''
    return dict(tenant_id=tenant_id, **extra)


def _make_pg_store(vector_settings):
    '''按 conftest 的 vector 分区构造同步 PG 向量库实例（未 start）。'''
    pg, pool = vector_settings.pg, vector_settings.pool_config
    dsn = os.environ.get('LOONG_TEST_VECTOR_PG_DSN', '') or pg.connection_string
    return PGVectorStore(
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
        connect_timeout=pg.connect_timeout,
    )


def _make_async_pg_store(vector_settings):
    '''按 conftest 的 vector 分区构造异步 PG 向量库实例（未 start）。'''
    pg, pool = vector_settings.pg, vector_settings.pool_config
    dsn = os.environ.get('LOONG_TEST_VECTOR_PG_DSN', '') or pg.connection_string
    return AsyncPGVectorStore(
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
        connect_timeout=pg.connect_timeout,
    )


# ---------- fixtures ----------


@pytest.fixture(scope='module')
def vector_settings():
    '''vector 分区配置：取 conftest 常量并深拷贝，避免用例间污染。'''
    return CONFIG_VECTOR_SETTINGS.model_copy(deep=True)


@pytest.fixture(scope='module')
def engine(vector_settings):
    '''同步向量引擎：start 内完成建表与索引；default_top_k 取 3 便于验证默认值。'''
    eng = VectorEngine(_make_pg_store(vector_settings), default_top_k=3)
    try:
        eng.start()
    except Exception as exc:
        pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
    yield eng

    store_engine = getattr(eng.store, 'engine', None)
    if store_engine is not None:
        store_engine.dispose()


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def async_engine(vector_settings):
    '''异步向量引擎：与同步用例共用同一集合，租户维度保证互不干扰。'''
    eng = AsyncVectorEngine(_make_async_pg_store(vector_settings), default_top_k=3)
    try:
        await eng.start()
    except Exception as exc:
        pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
    yield eng

    store_engine = getattr(eng.store, 'engine', None)
    if store_engine is not None:
        await store_engine.dispose()


# ---------- 契约层：构造参数与属性 ----------


class TestVectorEngineContract:
    '''构造契约与属性读取：只构造实例，不建立任何数据库连接。'''

    def test_backend_reads_store_backend(self):
        assert VectorEngine(PGVectorStore()).backend == 'pg'

    def test_collection_name_reads_store_value(self):
        store = PGVectorStore(collection_name=TEST_COLLECTION)
        assert VectorEngine(store).collection_name == TEST_COLLECTION

    def test_vector_size_reads_store_value(self):
        store = PGVectorStore(vector_size=TEST_VECTOR_SIZE)
        assert VectorEngine(store).vector_size == TEST_VECTOR_SIZE

    def test_default_top_k_when_not_given(self):
        assert VectorEngine(PGVectorStore()).default_top_k == 5

    def test_init_rejects_async_store(self):
        with pytest.raises(StorageError):
            VectorEngine(AsyncPGVectorStore())

    def test_async_init_rejects_sync_store(self):
        with pytest.raises(StorageError):
            AsyncVectorEngine(PGVectorStore())

    def test_enter_returns_self(self):
        eng = VectorEngine(PGVectorStore())
        assert eng.__enter__() is eng

    def test_exit_returns_none(self):
        eng = VectorEngine(PGVectorStore())
        assert eng.__exit__(None, None, None) is None


# ---------- 真实服务：同步引擎 ----------


class TestVectorEngineIntegration:
    '''VectorEngine 在真实 PostgreSQL 向量库上的公共方法行为。'''

    def test_start_starts_underlying_store(self, vector_settings):
        '''start 不返回自身，只负责拉起底层连接与建表。'''
        eng = VectorEngine(_make_pg_store(vector_settings), default_top_k=3)
        assert eng.start() is None
        try:
            assert eng.store.engine is not None
        finally:
            eng.store.engine.dispose()

    def test_insert_returns_written_count(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        count = engine.insert(tenant_id, [_vector(_random_seed())],
                             payloads=[_payload(tenant_id, None)], ids=[rid])
        assert count == 1
        assert engine.get(rid, tenant_id) is not None

    def test_insert_empty_vectors_returns_zero(self, engine):
        assert engine.insert(_next_id(), [], payloads=[], ids=[]) == 0

    def test_insert_rejects_dimension_mismatch(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        with pytest.raises(StorageError):
            engine.insert(tenant_id, [[0.1, 0.2]],
                          payloads=[_payload(tenant_id, None)], ids=[rid])

    def test_insert_rejects_payload_count_mismatch(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        with pytest.raises(StorageError):
            engine.insert(tenant_id, [_vector(_random_seed())], payloads=[], ids=[rid])

    def test_insert_rejects_ids_count_mismatch(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        with pytest.raises(StorageError):
            engine.insert(tenant_id, [_vector(_random_seed())],
                          payloads=[_payload(tenant_id, None)], ids=[rid, _next_id()])

    def test_get_returns_inserted_record(self, engine):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text = _random_text()
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, user_id, text=text)], ids=[rid])
        got = engine.get(rid, tenant_id, user_id)
        assert got is not None
        assert (got.id, got.tenant_id, got.user_id) == (rid, tenant_id, user_id)
        assert got.text == text

    def test_get_unknown_id_returns_none(self, engine):
        assert engine.get(_next_id(), _next_id()) is None

    def test_get_with_other_tenant_returns_none(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, None)], ids=[rid])
        assert engine.get(rid, _next_id()) is None

    def test_search_returns_hits_ordered_by_score_desc(self, engine):
        '''查询向量与第一条完全一致，故命中首位且 score 降序排列。'''
        tenant_id = _next_id()
        seeds = [_random_seed() for _ in range(3)]
        ids = [_next_id() for _ in range(3)]
        engine.insert(tenant_id, [_vector(seed) for seed in seeds],
                      payloads=[_payload(tenant_id, None) for _ in ids], ids=ids)
        hits = engine.search('query', _vector(seeds[0]), tenant_id,
                             top_k=3, filters=_scope(tenant_id))
        assert [hit.id for hit in hits][0] == ids[0]
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, engine):
        tenant_id = _next_id()
        ids = [_next_id() for _ in range(5)]
        engine.insert(tenant_id, [_vector(_random_seed()) for _ in ids],
                      payloads=[_payload(tenant_id, None) for _ in ids], ids=ids)
        assert len(engine.search('query', _vector(_random_seed()), tenant_id, top_k=2)) == 2

    def test_search_uses_default_top_k_when_none(self, engine):
        '''top_k 缺省时取 default_top_k（fixture 取 3）。'''
        tenant_id = _next_id()
        ids = [_next_id() for _ in range(5)]
        engine.insert(tenant_id, [_vector(_random_seed()) for _ in ids],
                      payloads=[_payload(tenant_id, None) for _ in ids], ids=ids)
        assert len(engine.search('query', _vector(_random_seed()), tenant_id)) == 3

    def test_search_filters_by_tenant(self, engine):
        tenant_a, tenant_b = _next_id(), _next_id()
        seed = _random_seed()
        rid_a, rid_b = _next_id(), _next_id()
        engine.insert(tenant_a, [_vector(seed)],
                      payloads=[_payload(tenant_a, None)], ids=[rid_a])
        engine.insert(tenant_b, [_vector(seed)],
                      payloads=[_payload(tenant_b, None)], ids=[rid_b])
        hits = engine.search('query', _vector(seed), tenant_a, top_k=5, filters=_scope(tenant_a))
        assert [hit.id for hit in hits] == [rid_a]

    def test_search_rejects_dimension_mismatch(self, engine):
        with pytest.raises(StorageError):
            engine.search('query', [0.1, 0.2], _next_id())

    def test_keyword_search_returns_matching_record(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        document = 'pgvector nearest neighbour index'
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, None, text=document)], ids=[rid])
        hits = engine.keyword_search('pgvector', tenant_id, top_k=10, filters=_scope(tenant_id))
        assert hits is not None
        assert [hit.id for hit in hits] == [rid]

    def test_keyword_search_returns_empty_for_absent_terms(self, engine):
        assert engine.keyword_search('missingterm', _next_id(), top_k=5) == []

    def test_list_returns_only_given_tenant(self, engine):
        '''必须传非空 filters：父类遇 filter=None 会另建字典，租户条件无法回写。'''
        tenant_a, tenant_b = _next_id(), _next_id()
        rid_a1, rid_a2, rid_b = _next_id(), _next_id(), _next_id()
        engine.insert(tenant_a, [_vector(_random_seed()) for _ in range(2)],
                      payloads=[_payload(tenant_a, None) for _ in range(2)], ids=[rid_a1, rid_a2])
        engine.insert(tenant_b, [_vector(_random_seed())],
                      payloads=[_payload(tenant_b, None)], ids=[rid_b])
        assert sorted(row.id for row in engine.list(tenant_a, filters=_scope(tenant_a))) == sorted([rid_a1, rid_a2])

    def test_list_without_filter_keeps_tenant_condition(self, engine):
        """不传 filters 时仍按租户隔离，只返回当前租户的数据。"""
        tenant_id = _next_id()
        rid = _next_id()
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, None)], ids=[rid])
        assert [row.id for row in engine.list(tenant_id)] == [rid]

    def test_list_respects_top_k(self, engine):
        tenant_id = _next_id()
        ids = [_next_id() for _ in range(4)]
        engine.insert(tenant_id, [_vector(_random_seed()) for _ in ids],
                      payloads=[_payload(tenant_id, None) for _ in ids], ids=ids)
        assert len(engine.list(tenant_id, filters=_scope(tenant_id), top_k=2)) == 2

    def test_update_changes_payload(self, engine):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        before, after = _distinct_texts(2)
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, user_id, text=before)], ids=[rid])
        assert engine.update(rid, tenant_id, user_id,
                            payload=_payload(tenant_id, user_id, text=after)) is None
        assert engine.get(rid, tenant_id, user_id).text == after

    def test_update_rejects_dimension_mismatch(self, engine):
        tenant_id, rid = _next_id(), _next_id()
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, None)], ids=[rid])
        with pytest.raises(StorageError):
            engine.update(rid, tenant_id, vector=[0.1, 0.2])

    def test_delete_removes_record(self, engine):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        engine.insert(tenant_id, [_vector(_random_seed())],
                      payloads=[_payload(tenant_id, user_id)], ids=[rid])
        assert engine.delete(rid, tenant_id, user_id) is None
        assert engine.get(rid, tenant_id, user_id) is None

    def test_delete_unknown_id_is_silent(self, engine):
        assert engine.delete(_next_id(), _next_id()) is None

    def test_get_by_filter_is_not_implemented_by_store(self, engine):
        '''已知缺陷：底层 store 未提供 get_by_filter，当前调用直接 AttributeError。'''
        with pytest.raises(AttributeError):
            engine.get_by_filter(_next_id(), _scope(_next_id()))


# ---------- 真实服务：异步引擎 ----------


@pytest.mark.asyncio(loop_scope='module')
class TestAsyncVectorEngineIntegration:
    '''AsyncVectorEngine 在真实 PostgreSQL 向量库上的公共方法行为。'''

    async def test_start_starts_underlying_store(self, vector_settings):
        eng = AsyncVectorEngine(_make_async_pg_store(vector_settings), default_top_k=3)
        assert await eng.start() is None
        try:
            assert eng.store.engine is not None
        finally:
            await eng.store.engine.dispose()

    async def test_insert_returns_written_count(self, async_engine):
        tenant_id, rid = _next_id(), _next_id()
        count = await async_engine.insert(tenant_id, [_vector(_random_seed())],
                                         payloads=[_payload(tenant_id, None)], ids=[rid])
        assert count == 1
        assert await async_engine.get(rid, tenant_id) is not None

    async def test_get_returns_inserted_record(self, async_engine):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        text = _random_text()
        await async_engine.insert(tenant_id, [_vector(_random_seed())],
                                  payloads=[_payload(tenant_id, user_id, text=text)], ids=[rid])
        got = await async_engine.get(rid, tenant_id, user_id)
        assert (got.id, got.tenant_id, got.user_id, got.text) == (rid, tenant_id, user_id, text)

    async def test_search_returns_hits_ordered_by_score_desc(self, async_engine):
        tenant_id = _next_id()
        seeds = [_random_seed() for _ in range(3)]
        ids = [_next_id() for _ in range(3)]
        await async_engine.insert(tenant_id, [_vector(seed) for seed in seeds],
                                  payloads=[_payload(tenant_id, None) for _ in ids], ids=ids)
        hits = await async_engine.search('query', _vector(seeds[0]), tenant_id,
                                        top_k=3, filters=_scope(tenant_id))
        scores = [hit.score for hit in hits]
        assert [hit.id for hit in hits][0] == ids[0]
        assert scores == sorted(scores, reverse=True)

    async def test_keyword_search_returns_matching_record(self, async_engine):
        tenant_id, rid = _next_id(), _next_id()
        document = 'snowflake identifiers keep primary keys unique'
        await async_engine.insert(tenant_id, [_vector(_random_seed())],
                                  payloads=[_payload(tenant_id, None, text=document)], ids=[rid])
        hits = await async_engine.keyword_search('snowflake', tenant_id, top_k=10,
                                                filters=_scope(tenant_id))
        assert [hit.id for hit in hits] == [rid]

    async def test_list_returns_only_given_tenant(self, async_engine):
        tenant_id = _next_id()
        ids = [_next_id() for _ in range(2)]
        await async_engine.insert(tenant_id, [_vector(_random_seed()) for _ in ids],
                                  payloads=[_payload(tenant_id, None) for _ in ids], ids=ids)
        assert sorted(row.id for row in await async_engine.list(tenant_id, filters=_scope(tenant_id))) == sorted(ids)

    async def test_update_changes_payload(self, async_engine):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        before, after = _distinct_texts(2)
        await async_engine.insert(tenant_id, [_vector(_random_seed())],
                                  payloads=[_payload(tenant_id, user_id, text=before)], ids=[rid])
        assert await async_engine.update(rid, tenant_id, user_id,
                                        payload=_payload(tenant_id, user_id, text=after)) is None
        assert (await async_engine.get(rid, tenant_id, user_id)).text == after

    async def test_delete_removes_record(self, async_engine):
        tenant_id, rid, user_id = _next_id(), _next_id(), _next_id()
        await async_engine.insert(tenant_id, [_vector(_random_seed())],
                                  payloads=[_payload(tenant_id, user_id)], ids=[rid])
        assert await async_engine.delete(rid, tenant_id, user_id) is None
        assert await async_engine.get(rid, tenant_id, user_id) is None


# ---------- 模块级构造入口 ----------


class TestVectorEngineFactories:
    '''build_vector_store、create_vector_engine、build_vector_engine 的行为。'''

    def test_build_vector_store_returns_pg_store(self):
        store = build_vector_store('pg', collection_name=TEST_COLLECTION,
                                   vector_size=TEST_VECTOR_SIZE)
        assert isinstance(store, PGVectorStore)
        assert (store.collection_name, store.vector_size) == (TEST_COLLECTION, TEST_VECTOR_SIZE)

    def test_build_vector_store_rejects_unknown_backend(self):
        with pytest.raises(ConfigError):
            build_vector_store('unknown')

    def test_create_vector_engine_wraps_store(self):
        eng = create_vector_engine('pg', default_top_k=7, collection_name=TEST_COLLECTION,
                                  vector_size=TEST_VECTOR_SIZE)
        assert isinstance(eng, VectorEngine)
        assert (eng.backend, eng.default_top_k) == ('pg', 7)

    def test_build_vector_engine_reads_settings(self, vector_settings):
        settings = vector_settings.model_copy(update={'backend': 'pg'})
        eng = build_vector_engine(settings)
        assert (eng.backend, eng.collection_name) == ('pg', settings.collection)
        assert (eng.vector_size, eng.default_top_k) == (settings.vector_size, settings.top_k)
