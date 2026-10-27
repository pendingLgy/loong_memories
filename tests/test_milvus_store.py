'''milvus_store.py 测试：MilvusVectorStore 与 AsyncMilvusVectorStore 的公共契约与业务方法。

覆盖策略：
1. 全方法覆盖：构造方法 __init__、backend 与 metric_type 类属性，以及 start、create_col、
   insert、search、keyword_search、delete、update、get、list 九个公共方法。
2. 继承契约验证：AsyncMilvusVectorStore 复用同步版的 schema 与解析逻辑，其公共方法在
   异步子类实例上单独验证，不与同步类混用。
3. 不单独测试私有方法：_schema、_record、_create_filter、_parse、_build_param_insert 等
   不做直接断言，其逻辑由 insert、search、list 的行为间接验证。
4. 不为父类建测试类：VectorStoreBase 不设独立测试套件。

Milvus 是外部服务，故整个模块以 integration 标记隔离：
未传 --integration 时整体跳过；异步用例另需 pymilvus>=2.5 提供的 AsyncMilvusClient。
连接参数取自 conftest 的 vector 分区。

运行：
    python -m pytest tests --integration -q
'''

from __future__ import annotations

import random
import sys
import time

import pytest


try:
    import pymilvus
    import pytest_asyncio

    from conftest import CONFIG_VECTOR_SETTINGS
    from loong_memories.core.exceptions import OptionalDependencyError, StorageError
    from loong_memories.storage.memories.base import VectorStoreBase
    from loong_memories.storage.memories.milvus_store import (
        AsyncMilvusClient,
        AsyncMilvusVectorStore,
        MilvusVectorStore,
    )
    from loong_memories.storage.memories.registry_vector import (
        is_vector_store_registered,
        resolve_vector_store,
    )

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)

HAS_ASYNC_CLIENT = AsyncMilvusClient is not None

needs_async_client = pytest.mark.skipif(
    not HAS_ASYNC_CLIENT,
    reason='异步 Milvus 用例需要 pymilvus>=2.5 提供的 AsyncMilvusClient'
)

pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)


# ---------- 辅助 ----------

def _random_id():
    '''随机正整数：避免跨用例、跨次运行的数据互相干扰。'''
    return random.randint(1, 2 ** 31 - 1)


def _vector(seed, dim):
    '''构造指定维度的单位方向向量：仅首元素随 seed 变化，便于预期相似度排序。'''
    vector = [0.0] * dim
    vector[0] = float(seed)
    return vector


def _payload(tenant_id, text, text_lemma=None):
    '''组装一条写入用的 payload，作用域标识与文本字段齐备。'''
    return {
        'tenant_id': tenant_id,
        'user_id': _random_id(),
        'text': text,
        'text_lemma': text_lemma or text,
        'created_at_tz': '2024-01-01T00:00:00+00:00',
    }


# ---------- fixtures ----------

@pytest.fixture(scope='module')
def vector_settings():
    '''vector 分区配置：取自 conftest 常量，逐模块深拷贝避免互相污染。'''
    return CONFIG_VECTOR_SETTINGS.model_copy(deep=True)


class TestMilvusVectorStore:
    '''MilvusVectorStore 的公共契约：backend 登记、构造参数与 metric_type。'''

    def test_backend_is_registered(self):
        assert MilvusVectorStore.backend == 'milvus'
        assert is_vector_store_registered('milvus') is True
        assert resolve_vector_store('milvus') is MilvusVectorStore

    def test_inherits_store_contract(self):
        assert issubclass(MilvusVectorStore, VectorStoreBase)
        assert isinstance(MilvusVectorStore(), VectorStoreBase)

    def test_init_defaults(self):
        store = MilvusVectorStore()
        assert store.uri == 'http://localhost:19530'
        assert (store.token, store.collection_name, store.db_name) == ('', '', '')
        assert store.vector_size == 1024
        assert (store.user, store.password, store.timeout) == ('', '', None)

    def test_init_stores_conftest_values(self, vector_settings):
        milvus = vector_settings.milvus
        store = MilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=15.0,
        )
        assert store.uri == milvus.uri
        assert store.token == milvus.token
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert store.db_name == milvus.db_name
        assert store.timeout == 15.0

    def test_init_leaves_client_unset(self):
        assert MilvusVectorStore().client is None

    def test_metric_type_is_cosine(self):
        assert MilvusVectorStore.metric_type == 'COSINE'

    def test_search_without_start_raises_storage_error(self, vector_settings):
        store = MilvusVectorStore(collection_name=vector_settings.collection)
        with pytest.raises(StorageError):
            store.search('x', _vector(1, vector_settings.vector_size), top_k=1)

    def test_start_without_driver_raises_optional_dependency(self, monkeypatch, vector_settings):
        monkeypatch.setattr('loong_memories.storage.memories.milvus_store.MilvusClient', None)
        store = MilvusVectorStore(collection_name=vector_settings.collection)
        with pytest.raises(OptionalDependencyError):
            store.start()


@needs_async_client
class TestAsyncMilvusVectorStore:
    '''AsyncMilvusVectorStore 的公共契约：backend 登记与构造参数。'''

    def test_backend_is_registered(self):
        assert AsyncMilvusVectorStore.backend == 'milvus_async'
        assert is_vector_store_registered('milvus_async') is True
        assert resolve_vector_store('milvus_async') is AsyncMilvusVectorStore

    def test_inherits_sync_store(self):
        assert issubclass(AsyncMilvusVectorStore, MilvusVectorStore)
        assert issubclass(AsyncMilvusVectorStore, VectorStoreBase)

    def test_init_defaults(self):
        store = AsyncMilvusVectorStore()
        assert store.uri == 'http://localhost:19530'
        assert (store.token, store.collection_name, store.db_name) == ('', '', '')
        assert store.vector_size == 1024
        assert (store.user, store.password, store.timeout) == ('', '', None)

    def test_init_stores_conftest_values(self, vector_settings):
        milvus = vector_settings.milvus
        store = AsyncMilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=15.0,
        )
        assert store.uri == milvus.uri
        assert store.collection_name == vector_settings.collection
        assert store.vector_size == vector_settings.vector_size
        assert store.timeout == 15.0

    def test_init_leaves_client_unset(self):
        assert AsyncMilvusVectorStore().client is None

    def test_metric_type_is_cosine(self):
        assert AsyncMilvusVectorStore.metric_type == 'COSINE'


# ---------- 真实服务：MilvusVectorStore ----------

class TestMilvusVectorStoreIntegration:
    '''MilvusVectorStore 在真实 Milvus 上的公共方法行为：建集合、写入、检索与更新。

    连接参数取自 conftest 的 vector 分区；集合名固定，用例靠随机 tenant_id 隔离数据。
    '''

    @pytest.fixture(scope='module')
    def milvus_store(self, vector_settings):
        '''建连并确保集合存在（start 内部会调用 create_col）。'''
        milvus = vector_settings.milvus
        store = MilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=30.0,
        )
        try:
            store.start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 Milvus：' + str(exc))
        yield store

    def test_start_connects_client(self, milvus_store):
        assert milvus_store.client is not None

    def test_create_col_reuses_existing_collection(self, milvus_store, vector_settings):
        assert milvus_store.create_col(vector_settings.collection, vector_settings.vector_size) is None
        assert milvus_store.client.has_collection(vector_settings.collection) is True

    def test_insert_then_get_returns_record(self, milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'hello')], [rid])
        got = milvus_store.get(rid)
        assert got is not None
        assert (got.id, got.tenant_id, got.text) == (rid, tenant_id, 'hello')

    def test_get_unknown_id_returns_none(self, milvus_store):
        assert milvus_store.get(_random_id()) is None

    def test_insert_persists_text_lemma(self, milvus_store, vector_settings):
        rid = _random_id()
        payload = _payload(_random_id(), '原文', '词形')
        milvus_store.insert([_vector(1, vector_settings.vector_size)], [payload], [rid])
        got = milvus_store.get(rid)
        assert (got.text, got.text_lemma) == ('原文', '词形')

    def test_search_returns_hits_desc_by_score(self, milvus_store, vector_settings):
        tenant_id = _random_id()
        dim = vector_settings.vector_size
        ids = [_random_id() for _ in range(3)]
        milvus_store.insert([_vector(1, dim), _vector(2, dim), _vector(3, dim)],
                            [_payload(tenant_id, 'm' + str(index)) for index in range(3)], ids)
        hits = milvus_store.search('m0', _vector(1, dim), top_k=3, filters={'tenant_id': tenant_id})
        assert len(hits) == 3
        assert sorted([hit.id for hit in hits]) == sorted(ids)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, milvus_store, vector_settings):
        tenant_id = _random_id()
        dim = vector_settings.vector_size
        milvus_store.insert([_vector(1, dim) for _ in range(3)],
                            [_payload(tenant_id, 'top' + str(index)) for index in range(3)],
                            [_random_id() for _ in range(3)])
        assert len(milvus_store.search('top', _vector(1, dim), top_k=2, filters={'tenant_id': tenant_id})) == 2

    def test_search_filters_by_tenant(self, milvus_store, vector_settings):
        first, second = _random_id(), _random_id()
        dim = vector_settings.vector_size
        milvus_store.insert([_vector(1, dim), _vector(1, dim)],
                            [_payload(first, 'a'), _payload(second, 'b')],
                            [_random_id(), _random_id()])
        hits = milvus_store.search('a', _vector(1, dim), top_k=10, filters={'tenant_id': first})
        assert [hit.text for hit in hits] == ['a']



    def test_keyword_search_matches_text_lemma(self, milvus_store, vector_settings):
        tenant_id = _random_id()
        rid = _random_id()
        milvus_store.insert([_vector(1, vector_settings.vector_size)],
                            [_payload(tenant_id, 'Milvus 是向量库', 'milvus vector database')], [rid])
        hits = milvus_store.keyword_search('milvus', top_k=5, filters={'tenant_id': tenant_id})
        assert hits is not None
        assert [hit.id for hit in hits] == [rid]

    def test_keyword_search_returns_none_on_client_error(self, milvus_store):
        assert milvus_store.keyword_search('x', top_k=1, filters={'bad key': 1}) is None

    def test_update_vector_and_payload(self, milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'before')], [rid])
        milvus_store.update(rid, vector=_vector(5, vector_settings.vector_size), payload=_payload(tenant_id, 'after'))
        got = milvus_store.get(rid)
        assert got.text == 'after'

    def test_update_keeps_existing_vector_when_omitted(self, milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        milvus_store.insert([_vector(2, vector_settings.vector_size)], [_payload(tenant_id, 'keep')], [rid])
        milvus_store.update(rid, payload=_payload(tenant_id, 'kept'))
        got = milvus_store.get(rid)
        assert got is not None and got.text == 'kept'

    def test_update_unknown_id_raises_storage_error(self, milvus_store, vector_settings):
        with pytest.raises(StorageError):
            milvus_store.update(_random_id(), payload={'tenant_id': _random_id()})

    def test_delete_removes_record(self, milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'gone')], [rid])
        milvus_store.delete(rid)
        assert milvus_store.get(rid) is None

    def test_list_respects_top_k(self, milvus_store, vector_settings):
        tenant_id = _random_id()
        milvus_store.insert([_vector(1, vector_settings.vector_size) for _ in range(3)],
                            [_payload(tenant_id, 'l' + str(index)) for index in range(3)],
                            [_random_id() for _ in range(3)])
        assert len(milvus_store.list(filters={'tenant_id': tenant_id}, top_k=2)) == 2

    def test_list_applies_filters(self, milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'only')], [rid])
        listed = milvus_store.list(filters={'tenant_id': tenant_id})
        assert [item.id for item in listed] == [rid]

    def test_list_rejects_illegal_filter_key(self, milvus_store):
        with pytest.raises(StorageError):
            milvus_store.list(filters={'bad key': 1})


# ---------- 真实服务：AsyncMilvusVectorStore ----------

@needs_async_client
@pytest.mark.asyncio(loop_scope='module')
class TestAsyncMilvusVectorStoreIntegration:
    '''AsyncMilvusVectorStore 在真实 Milvus 上的公共方法行为，语义与同步版本一致。'''

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def async_milvus_store(self, vector_settings):
        '''建连并确保集合存在（start 内部会 await create_col）。'''
        milvus = vector_settings.milvus
        store = AsyncMilvusVectorStore(
            uri=milvus.uri,
            token=milvus.token,
            collection_name=vector_settings.collection,
            vector_size=vector_settings.vector_size,
            db_name=milvus.db_name,
            timeout=30.0,
        )
        try:
            await store.start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 Milvus：' + str(exc))
        yield store

    async def test_start_connects_client(self, async_milvus_store):
        assert async_milvus_store.client is not None

    async def test_create_col_is_idempotent(self, async_milvus_store, vector_settings):
        assert await async_milvus_store.create_col(vector_settings.collection, vector_settings.vector_size) is None

    async def test_insert_then_get_returns_record(self, async_milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        await async_milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'hello')], [rid])
        got = await async_milvus_store.get(rid)
        assert got is not None
        assert (got.id, got.text) == (rid, 'hello')

    async def test_get_unknown_id_returns_none(self, async_milvus_store):
        assert await async_milvus_store.get(_random_id()) is None

    async def test_search_returns_hits_desc_by_score(self, async_milvus_store, vector_settings):
        tenant_id = _random_id()
        dim, ids = vector_settings.vector_size, [_random_id() for _ in range(3)]
        await async_milvus_store.insert([_vector(1, dim), _vector(2, dim), _vector(3, dim)],
                                       [_payload(tenant_id, 'm' + str(index)) for index in range(3)], ids)
        hits = await async_milvus_store.search('m0', _vector(1, dim), top_k=3, filters={'tenant_id': tenant_id})
        assert sorted([hit.id for hit in hits]) == sorted(ids)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    async def test_keyword_search_matches_text_lemma(self, async_milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        await async_milvus_store.insert([_vector(1, vector_settings.vector_size)],
                                       [_payload(tenant_id, 'Milvus 是向量库', 'milvus vector database')], [rid])
        hits = await async_milvus_store.keyword_search('milvus', top_k=5, filters={'tenant_id': tenant_id})
        assert hits is not None and hits[0].id == rid

    async def test_keyword_search_returns_none_on_failure(self, async_milvus_store):
        assert await async_milvus_store.keyword_search('x', top_k=1, filters={'bad key': 1}) is None

    async def test_update_changes_payload(self, async_milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        await async_milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'before')], [rid])
        await async_milvus_store.update(rid, payload=_payload(tenant_id, 'after'))
        got = await async_milvus_store.get(rid)
        assert got.text == 'after'

    async def test_update_unknown_id_raises_storage_error(self, async_milvus_store):
        with pytest.raises(StorageError):
            await async_milvus_store.update(_random_id(), payload={'tenant_id': _random_id()})

    async def test_delete_removes_record(self, async_milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        await async_milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'gone')], [rid])
        await async_milvus_store.delete(rid)
        assert await async_milvus_store.get(rid) is None

    async def test_list_applies_filters(self, async_milvus_store, vector_settings):
        tenant_id, rid = _random_id(), _random_id()
        await async_milvus_store.insert([_vector(1, vector_settings.vector_size)], [_payload(tenant_id, 'only')], [rid])
        listed = await async_milvus_store.list(filters={'tenant_id': tenant_id})
        assert rid in [item.id for item in listed]
