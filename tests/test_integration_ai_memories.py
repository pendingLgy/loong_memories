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

'''
loong_memories 统一入口 memories 测试

覆盖五部分：
1. 公开入口契约：Memories 与 AsyncMemories 的 load 与 create 签名、类属性与导出清单。
2. 实现族判定：_db_is_async、_engine_is_async、_store_is_async。
3. 装配辅助：_db_kwargs、_vector_kwargs、_check_settings、_check_family、_store_class、_build_vector。
4. 装配编排：_assemble 的配置驱动构造、三层族校验与 nlp 与 rerank 分区透传。
5. 真实负载集成：Memories 与 AsyncMemories 的 load 与 start，需 --integration_ai。

依赖缺失时整体跳过而不是报错：memories 顶层导入 memory.handler（需 sqlalchemy）
与 core.config_memory（需 pydantic-settings），两者任一缺失都会在导入期失败。

运行：
    python -m pytest tests -q -k memories
    python -m pytest tests -q -k memories --integration_ai
'''

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import random
import sys

import pytest

try:
    import pytest_asyncio
    from dotenv import load_dotenv
    from sqlalchemy import text

    from conftest import (
        CONFIG_DB,
        CONFIG_DB_SETTINGS,
        CONFIG_JSON_DICT,
        CONFIG_VECTOR,
        CONFIG_VECTOR_SETTINGS,
    )
    from loong_memories.core.config_db import DBSettings
    from loong_memories.core.config_memory import MemorySettings
    from loong_memories.core.config_vector import VectorSettings
    from loong_memories.core.exceptions import ConfigError
    from loong_memories.memory.handler import AsyncMemoryHandler, MemoryHandler
    from loong_memories.memory.models import (
        AddResult,
        DeleteResult,
        GetResult,
        PageQueryResult,
        SearchResult,
        UpdateResult,
    )
    from loong_memories.memories import (
        REQUIRED_PARTITIONS,
        AsyncMemories,
        Memories,
        MemoriesBase,
        _assemble,
        _build_vector,
        _check_family,
        _check_settings,
        _db_is_async,
        _db_kwargs,
        _engine_is_async,
        _store_class,
        _store_is_async,
        _vector_kwargs,
    )
    from loong_memories.storage.memories.vector_engine import AsyncVectorEngine, VectorEngine
    from loong_memories.storage.message.sqlite_db import AsyncSQLiteDB, SQLiteDB
except ImportError as _exc:
    pytest.skip('[SKIP INFO] File ' + __file__ + '：缺少依赖 ' + str(_exc), allow_module_level=True)


# 集成用例默认跳过：需要真实数据库、向量库与 LLM 凭据，用 --integration_ai 显式开启。
pytestmark = pytest.mark.integration_ai

if  '--integration_ai' not in sys.argv:
    pytest.skip(f"[SKIP INFO] File {__file__}：integration_ai 集成测试，默认跳过（请使用 --integration_ai 参数运行）",
                allow_module_level=True)

# 族校验报错文案里统一使用的装配方名。
logger = logging.getLogger(__name__)


# 显式加载环境变量 .env：真实模型的密钥与地址从中读取。
load_dotenv()


LABEL = 'Memories'


def _assemble_sync(settings):
    '''以同步入口身份调用装配：与 Memories.create 的调用形态一致。'''
    return _assemble(settings, is_async=False, handler_cls=MemoryHandler, label=LABEL)


def _assemble_async(settings):
    '''以异步入口身份调用装配：与 AsyncMemories.create 的调用形态一致。'''
    return _assemble(settings, is_async=True, handler_cls=AsyncMemoryHandler, label=LABEL)


class _FakeStore:
    '''向量库替身：同步检索方法，供族判定与装配编排使用。'''

    backend = 'fake'

    def search(self, query, tenant_id, user_id=None, top_k=None, filter=None):
        return []


class _FakeAsyncStore:
    '''向量库替身：异步检索方法。'''

    backend = 'fake_async'

    async def search(self, query, tenant_id, user_id=None, top_k=None, filter=None):
        return []


class _FakeDB:
    '''关系库替身：只声明 backend，不建立连接。'''

    backend = 'sqlite'


def _settings(**overrides) -> MemorySettings:
    '''由 conftest 的完整配置构造 MemorySettings，overrides 按分区整体替换。'''
    data = copy.deepcopy(CONFIG_JSON_DICT)
    # conftest 的组合配置不含顶层总开关，门面要求 enable=true，这里统一补上。
    data['enable'] = True
    data.update(overrides)
    return MemorySettings.model_validate(data)

@pytest.fixture
def memory_settings() -> MemorySettings:
    '''同步后端完整配置：sqlite 与 milvus。'''
    return _settings()


@pytest.fixture
def async_memory_settings() -> MemorySettings:
    '''异步后端完整配置：sqlite_async 与 milvus_async。'''
    db = copy.deepcopy(CONFIG_DB)
    db['backend'] = 'sqlite_async'
    vector = copy.deepcopy(CONFIG_VECTOR)
    vector['backend'] = 'milvus_async'
    return _settings(db=db, vector=vector)


class TestPublicApiContract:
    '''公开入口契约：签名、类属性与导出清单。'''

    def test_all_exports_expected_names(self):
        import loong_memories.memories as module

        assert module.__all__ == [
            'AsyncMemories',
            'Memories',
            'MemoriesBase',
            'REQUIRED_PARTITIONS',
        ]

    def test_required_partitions_cover_four_services(self):
        assert set(REQUIRED_PARTITIONS) == {
            'embedding', 'llm', 'db', 'vector',
        }

    @pytest.mark.parametrize('cls', [Memories, AsyncMemories])
    def test_create_takes_settings_only(self, cls):
        params = list(inspect.signature(cls.create).parameters.values())
        assert [param.name for param in params] == ['settings']

    @pytest.mark.parametrize('cls', [Memories, AsyncMemories])
    def test_load_takes_config_sources_only(self, cls):
        params = list(inspect.signature(cls.load).parameters.values())
        assert [param.name for param in params] == [
            'json_text',
            'yaml_path',
            'use_env',
            'env_prefix',
        ]

    @pytest.mark.parametrize('cls', [Memories, AsyncMemories])
    def test_neither_entry_accepts_component_injection(self, cls):
        for method in (cls.load, cls.create):
            names = list(inspect.signature(method).parameters)
            for forbidden in ('embedder', 'llm', 'reranker', 'db', 'vector'):
                assert forbidden not in names

    def test_family_attributes_of_sync_entry(self):
        assert Memories.is_async is False
        assert Memories.handler_cls is MemoryHandler

    def test_family_attributes_of_async_entry(self):
        assert AsyncMemories.is_async is True
        assert AsyncMemories.handler_cls is AsyncMemoryHandler

    @pytest.mark.parametrize('cls', [Memories, AsyncMemories])
    def test_both_entries_inherit_base(self, cls):
        assert issubclass(cls, MemoriesBase)

    def test_helpers_are_module_level_private_functions(self):
        import loong_memories.memories as module

        helpers = [
            name
            for name, value in vars(module).items()
            if inspect.isfunction(value) and value.__module__ == module.__name__
        ]
        assert helpers
        assert all(name.startswith('_') for name in helpers)

    def test_base_keeps_only_entry_points(self):
        members = sorted(name for name in vars(MemoriesBase) if not name.startswith('__'))
        assert members == ['create', 'handler_cls', 'is_async', 'load']


class TestAsyncFamilyDetection:
    '''实现族判定：一律看实现类，不看 backend 名后缀。'''

    def test_db_is_async_reads_engine_class(self):
        assert _db_is_async(SQLiteDB.__new__(SQLiteDB)) is False
        assert _db_is_async(AsyncSQLiteDB.__new__(AsyncSQLiteDB)) is True

    def test_db_is_async_treats_missing_db_as_sync(self):
        assert _db_is_async(None) is False

    def test_engine_is_async_reads_engine_class(self):
        assert _engine_is_async(VectorEngine(_FakeStore())) is False
        assert _engine_is_async(AsyncVectorEngine(_FakeAsyncStore())) is True

    def test_store_is_async_reads_search_signature(self):
        assert _store_is_async(_FakeStore()) is False
        assert _store_is_async(_FakeAsyncStore()) is True

    def test_store_without_search_counts_as_sync(self):
        assert _store_is_async(object()) is False

    def test_async_backend_name_does_not_decide_store_family(self):
        store = _FakeStore()
        store.backend = 'pg_async'
        assert _store_is_async(store) is False

    def test_sync_engine_accepts_async_backend_name(self):
        store = _FakeStore()
        store.backend = 'pg_async'
        assert _engine_is_async(VectorEngine(store)) is False


class TestDbKwargs:
    '''关系库连接参数按后端组装，连接池与表前缀通用于所有方言。'''

    def test_common_pool_keys_present(self):
        kwargs = _db_kwargs(CONFIG_DB_SETTINGS)
        for key in (
            'print_sql',
            'pool_size',
            'max_overflow',
            'pool_timeout',
            'pool_recycle',
            'pool_pre_ping',
            'table_prefix',
        ):
            assert key in kwargs

    def test_sqlite_branch(self):
        kwargs = _db_kwargs(CONFIG_DB_SETTINGS)
        assert kwargs['db_path'] == CONFIG_DB_SETTINGS.sqlite.db_path
        assert kwargs['timeout'] == CONFIG_DB_SETTINGS.sqlite.timeout
        assert kwargs['check_same_thread'] is CONFIG_DB_SETTINGS.sqlite.check_same_thread
        assert 'host' not in kwargs

    def test_sqlite_requires_db_path(self):
        db = copy.deepcopy(CONFIG_DB)
        db['sqlite']['db_path'] = ''
        with pytest.raises(ConfigError):
            _db_kwargs(DBSettings.model_validate(db))

    def test_mysql_branch(self):
        db = copy.deepcopy(CONFIG_DB)
        db['backend'] = 'mysql'
        kwargs = _db_kwargs(DBSettings.model_validate(db))
        assert kwargs['host'] == db['mysql']['host']
        assert kwargs['charset'] == db['mysql']['charset']
        assert 'db_path' not in kwargs

    def test_pg_branch(self):
        db = copy.deepcopy(CONFIG_DB)
        db['backend'] = 'pg'
        kwargs = _db_kwargs(DBSettings.model_validate(db))
        assert kwargs['sslmode'] == db['pg']['sslmode']
        assert kwargs['application_name'] == db['pg']['application_name']
        assert kwargs['target_session_attrs'] == db['pg']['target_session_attrs']
        assert 'charset' not in kwargs

    def test_async_backend_shares_prefix_with_sync(self):
        db = copy.deepcopy(CONFIG_DB)
        db['backend'] = 'sqlite_async'
        kwargs = _db_kwargs(DBSettings.model_validate(db))
        assert kwargs['db_path'] == db['sqlite']['db_path']


class TestVectorKwargs:
    '''向量库连接参数按实现类族组装，空字符串不下传。'''

    def test_milvus_branch(self):
        kwargs = _vector_kwargs(CONFIG_VECTOR_SETTINGS)
        assert set(kwargs) <= {'uri', 'token', 'db_name'}
        assert kwargs['uri'] == CONFIG_VECTOR_SETTINGS.milvus.uri

    def test_milvus_async_does_not_fall_into_pg_branch(self):
        vector = copy.deepcopy(CONFIG_VECTOR)
        vector['backend'] = 'milvus_async'
        kwargs = _vector_kwargs(VectorSettings.model_validate(vector))
        assert kwargs['db_name'] == CONFIG_VECTOR_SETTINGS.milvus.db_name
        assert 'host' not in kwargs
        assert 'use_hnsw' not in kwargs

    def test_pg_branch(self):
        vector = copy.deepcopy(CONFIG_VECTOR)
        vector['backend'] = 'pg'
        kwargs = _vector_kwargs(VectorSettings.model_validate(vector))
        assert kwargs['host'] == CONFIG_VECTOR_SETTINGS.pg.host
        assert kwargs['port'] == CONFIG_VECTOR_SETTINGS.pg.port
        assert kwargs['use_hnsw'] is CONFIG_VECTOR_SETTINGS.pg.use_hnsw
        assert 'uri' not in kwargs

    def test_empty_string_fields_are_dropped(self):
        vector = copy.deepcopy(CONFIG_VECTOR)
        vector['backend'] = 'pg'
        vector['pg']['connection_string'] = ''
        vector['pg']['password'] = ''
        kwargs = _vector_kwargs(VectorSettings.model_validate(vector))
        assert 'connection_string' not in kwargs
        assert 'password' not in kwargs

    def test_false_boolean_field_is_kept(self):
        vector = copy.deepcopy(CONFIG_VECTOR)
        vector['backend'] = 'pg'
        vector['pg']['use_diskann'] = False
        kwargs = _vector_kwargs(VectorSettings.model_validate(vector))
        assert kwargs['use_diskann'] is False


class TestCheckSettings:
    '''总开关与四个硬依赖分区的门控。'''

    def test_rejects_disabled_master_switch(self, memory_settings):
        memory_settings.enable = False
        with pytest.raises(ConfigError):
            _check_settings(memory_settings, LABEL)

    @pytest.mark.parametrize('partition', ['embedding', 'llm', 'db', 'vector'])
    def test_rejects_disabled_required_partition(self, memory_settings, partition):
        getattr(memory_settings, partition).enable = False
        with pytest.raises(ConfigError) as excinfo:
            _check_settings(memory_settings, LABEL)
        assert partition in str(excinfo.value)

    def test_accepts_all_required_partitions_enabled(self, memory_settings):
        assert _check_settings(memory_settings, LABEL) is None

    def test_optional_partitions_may_stay_disabled(self, memory_settings):
        memory_settings.rerank.enable = False
        memory_settings.nlp.enable = False
        assert _check_settings(memory_settings, LABEL) is None

    def test_family_not_checked_from_backend_name(self):
        db = copy.deepcopy(CONFIG_DB)
        db['backend'] = 'sqlite_async'
        assert _check_settings(_settings(db=db), LABEL) is None


class TestCheckFamily:
    '''族不一致时报错，文案给出实际类名与目标族。'''

    def test_sync_role_accepts_sync_impl(self):
        assert _check_family(LABEL, '关系库', 'SQLiteDB', False, False) is None

    def test_async_role_accepts_async_impl(self):
        assert _check_family(LABEL, '关系库', 'AsyncSQLiteDB', True, True) is None

    def test_sync_role_rejects_async_impl(self):
        with pytest.raises(ConfigError) as excinfo:
            _check_family(LABEL, '关系库', 'AsyncSQLiteDB', True, False)
        message = str(excinfo.value)
        assert 'AsyncSQLiteDB' in message
        assert '同步' in message

    def test_async_role_rejects_sync_impl(self):
        with pytest.raises(ConfigError) as excinfo:
            _check_family(LABEL, '向量引擎', 'VectorEngine', False, True)
        message = str(excinfo.value)
        assert 'VectorEngine' in message
        assert '异步' in message


class TestStoreClassResolution:
    '''向量库实现类解析：同步直取，异步在同模块内找协程子类。'''

    def test_unknown_backend_raises_config_error(self):
        with pytest.raises(ConfigError):
            _store_class('unknown_backend', False)

    def test_sync_backend_returns_sync_class(self):
        pytest.importorskip('pgvector')
        from loong_memories.storage.memories.pg_store import PGVectorStore

        assert _store_class('pg', False) is PGVectorStore

    def test_async_backend_returns_async_subclass(self):
        pytest.importorskip('pgvector')
        from loong_memories.storage.memories.pg_store import AsyncPGVectorStore, PGVectorStore

        resolved = _store_class('pg', True)
        assert resolved is AsyncPGVectorStore
        assert issubclass(resolved, PGVectorStore)
        assert _store_is_async(resolved) is True

    def test_async_suffix_backend_normalized_to_base(self):
        pytest.importorskip('pgvector')
        from loong_memories.storage.memories.pg_store import AsyncPGVectorStore, PGVectorStore

        assert _store_class('pg_async', False) is PGVectorStore
        assert _store_class('pg_async', True) is AsyncPGVectorStore

    def test_milvus_async_backend_returns_async_subclass(self):
        pytest.importorskip('pymilvus')
        from loong_memories.storage.memories.milvus_store import AsyncMilvusVectorStore

        assert _store_class('milvus', True) is AsyncMilvusVectorStore
        assert _store_class('milvus_async', True) is AsyncMilvusVectorStore

    def test_async_backend_without_async_impl_raises(self, monkeypatch):
        import loong_memories.memories as module

        class _OnlySyncStore:
            def search(self, query):
                return []

        monkeypatch.setattr(module, 'resolve_vector_store', lambda name: _OnlySyncStore)
        with pytest.raises(ConfigError):
            _store_class('milvus', True)


class TestBuildVector:
    '''向量引擎构造：集合名必填，维度与 top_k 注入引擎。'''

    def test_empty_collection_raises_config_error(self):
        vector = copy.deepcopy(CONFIG_VECTOR)
        vector['collection'] = ''
        with pytest.raises(ConfigError):
            _build_vector(VectorSettings.model_validate(vector), False)

    def test_sync_engine_wraps_sync_store(self):
        pytest.importorskip('pymilvus')
        engine = _build_vector(CONFIG_VECTOR_SETTINGS, False)
        assert isinstance(engine, VectorEngine)
        assert engine.default_top_k == CONFIG_VECTOR_SETTINGS.top_k
        assert engine.vector_size == CONFIG_VECTOR_SETTINGS.vector_size
        assert engine.collection_name == CONFIG_VECTOR_SETTINGS.collection
        assert _store_is_async(engine.store) is False

    def test_async_engine_wraps_async_store(self):
        pytest.importorskip('pymilvus')
        vector = copy.deepcopy(CONFIG_VECTOR)
        vector['backend'] = 'milvus_async'
        settings = VectorSettings.model_validate(vector)
        engine = _build_vector(settings, True)
        assert isinstance(engine, AsyncVectorEngine)
        assert _store_is_async(engine.store) is True
        assert engine.default_top_k == settings.top_k


class TestAssemble:
    '''装配编排：配置驱动构造、三层族校验与可选分区透传。'''

    @pytest.fixture
    def patched(self, monkeypatch):
        '''替换底层构造：关系库与向量引擎改由替身产出，独立验证装配编排。'''
        import loong_memories.memories as module

        state = {
            'db': _FakeDB(),
            'db_backend': None,
            'db_kwargs': {},
            'vector_async': None,
        }

        def fake_resolve_db(name):
            state['db_backend'] = name

            def factory(**kwargs):
                state['db_kwargs'] = kwargs
                return state['db']

            return factory

        def fake_build_vector(settings, is_async):
            state['vector_async'] = is_async
            if is_async:
                return AsyncVectorEngine(_FakeAsyncStore())
            return VectorEngine(_FakeStore())

        monkeypatch.setattr(module, 'resolve_db', fake_resolve_db)
        monkeypatch.setattr(module, '_build_vector', fake_build_vector)
        return state

    def test_returns_sync_handler(self, memory_settings, patched):
        handler = _assemble_sync(memory_settings)
        assert isinstance(handler, MemoryHandler)
        assert patched['db_backend'] == 'sqlite'
        assert patched['vector_async'] is False
        assert patched['db_kwargs']['table_prefix'] == memory_settings.db.table_prefix

    def test_returns_async_handler(self, async_memory_settings, patched):
        patched['db'] = AsyncSQLiteDB.__new__(AsyncSQLiteDB)
        handler = _assemble_async(async_memory_settings)
        assert isinstance(handler, AsyncMemoryHandler)
        assert patched['db_backend'] == 'sqlite_async'
        assert patched['vector_async'] is True

    def test_nlp_disabled_passes_none(self, memory_settings, patched):
        memory_settings.nlp.enable = False
        handler = _assemble_sync(memory_settings)
        assert handler.nlp_setting is None

    def test_nlp_enabled_passes_settings(self, memory_settings, patched):
        memory_settings.nlp.enable = True
        handler = _assemble_sync(memory_settings)
        assert handler.nlp_setting is not None
        assert handler.nlp_setting.enable is True

    def test_rerank_disabled_passes_none(self, memory_settings, patched):
        memory_settings.rerank.enable = False
        handler = _assemble_sync(memory_settings)
        assert handler.reranker is None

    def test_rerank_enabled_passes_handler(self, memory_settings, patched):
        memory_settings.rerank.enable = True
        handler = _assemble_sync(memory_settings)
        assert handler.reranker is not None

    def test_rejects_sync_entry_with_async_db(self, memory_settings, patched, monkeypatch):
        import loong_memories.memories as module

        async_db = AsyncSQLiteDB.__new__(AsyncSQLiteDB)
        monkeypatch.setattr(module, 'resolve_db', lambda name: (lambda **kwargs: async_db))
        with pytest.raises(ConfigError) as excinfo:
            _assemble_sync(memory_settings)
        assert '关系库' in str(excinfo.value)

    def test_rejects_async_entry_with_sync_db(self, async_memory_settings, patched, monkeypatch):
        import loong_memories.memories as module

        monkeypatch.setattr(module, 'resolve_db', lambda name: (lambda **kwargs: _FakeDB()))
        with pytest.raises(ConfigError) as excinfo:
            _assemble_async(async_memory_settings)
        assert '关系库' in str(excinfo.value)

    def test_rejects_sync_entry_wrapped_async_store(self, memory_settings, patched, monkeypatch):
        import loong_memories.memories as module

        engine = VectorEngine(_FakeStore())
        engine._store_base = _FakeAsyncStore()
        monkeypatch.setattr(module, '_build_vector', lambda settings, is_async: engine)
        with pytest.raises(ConfigError) as excinfo:
            _assemble_sync(memory_settings)
        assert '向量库实现' in str(excinfo.value)

    def test_rejects_engine_without_store(self, memory_settings, patched, monkeypatch):
        import loong_memories.memories as module

        monkeypatch.setattr(module, '_build_vector', lambda settings, is_async: object())
        with pytest.raises(ConfigError) as excinfo:
            _assemble_sync(memory_settings)
        assert '向量引擎' in str(excinfo.value)


class TestEntryFamilyGuards:
    '''入口与配置族不一致时在装配期报错。'''

    def test_sync_entry_rejects_async_db_config(self):
        pytest.importorskip('pymilvus')
        db = copy.deepcopy(CONFIG_DB)
        db['backend'] = 'sqlite_async'
        with pytest.raises(ConfigError):
            Memories.create(_settings(db=db))

    def test_async_entry_rejects_sync_config(self):
        pytest.importorskip('pymilvus')
        with pytest.raises(ConfigError):
            AsyncMemories.create(_settings())

    def test_async_entry_accepts_async_config(self, monkeypatch):
        import loong_memories.memories as module

        monkeypatch.setattr(
            module, 'resolve_db',
            lambda name: (lambda **kwargs: AsyncSQLiteDB.__new__(AsyncSQLiteDB)),
        )
        monkeypatch.setattr(
            module, '_build_vector',
            lambda settings, is_async: AsyncVectorEngine(_FakeAsyncStore()),
        )
        handler = AsyncMemories.create(_settings())
        assert isinstance(handler, AsyncMemoryHandler)


# ---------- 第三部分：真实关系库与向量库集成（需 --integration_ai） ----------

MAX_ENTITY_ID = 2 ** 31 - 1


def random_identity() -> tuple:
    '''随机租户与用户：每次调用产生新的隔离边界，避免用例之间串数据。'''
    return random.randint(1, MAX_ENTITY_ID), random.randint(1, MAX_ENTITY_ID)


def generate_extractable_human_dialogue() -> tuple:
    '''生成人类真实对话测试数据，返回 (messages, search_keywords, theme)。'''
    templates = [
        {
            'theme': '技术栈与偏好',
            'keywords': ['uv', 'pgvector', 'PyCharm', '美式咖啡'],
            'messages': [
                {'role': 'user', 'content': '我最近把开发环境全面转到 uv 和 Hatch 了，比原来清爽太多。'},
                {'role': 'assistant', 'content': 'uv 的虚拟环境与依赖解析速度很快，你的主项目还在做 pgvector 迁移吗？'},
                {'role': 'user', 'content': '对，不仅在做 pgvector，还顺手把 LangGraph 的工作流重构了一遍。'},
                {'role': 'assistant', 'content': '长期记忆模块你准备自己写还是用现成方案？'},
                {'role': 'user', 'content': '我打算自己定制一层状态压缩，另外我习惯喝美式咖啡来保持深夜写代码的清醒。'},
            ],
        },
        {
            'theme': '工作日常与生活节奏',
            'keywords': ['数据优化', '周末爬山', 'pyautogui', '深夜写代码'],
            'messages': [
                {'role': 'user', 'content': '今天又是高强度的一天，处理了三百万级并发的数据优化。'},
                {'role': 'assistant', 'content': '听起来是大工程，你不是一向喜欢在安静的深夜处理核心架构问题吗？'},
                {'role': 'user', 'content': '对，我通常晚上十点以后灵感最好，不过周末一定会抽半天去爬山。'},
                {'role': 'assistant', 'content': '劳逸结合很关键。那个自动化 GUI 助手项目进展如何？'},
                {'role': 'user', 'content': '用 Python 与 pyautogui 写的桌面 Agent 已经跑通了，准备加网页数据抓取。'},
            ],
        },
    ]
    selected = random.choice(templates)
    return selected['messages'], selected['keywords'], selected['theme']


def _settings_data(db_backend, vector_backend, table_prefix, collection) -> dict:
    '''按后端与资源名组装完整配置：模型凭据取自 .env，其余分区取自 conftest。'''
    data = copy.deepcopy(CONFIG_JSON_DICT)
    # conftest 的组合配置不含顶层总开关，门面要求 enable=true，这里统一补上。
    data['enable'] = True
    data['db']['backend'] = db_backend
    data['db']['table_prefix'] = table_prefix
    data['vector']['backend'] = vector_backend
    data['vector']['collection'] = collection
    # 三类模型的凭据均取自 .env：逐分区显式读取，变量名一目了然。
    data['embedding'].update({
        'provider': 'qwen',
        'model': os.getenv('embedding_model', ''),
        'api_key': os.getenv('embedding_api_key', ''),
        'base_url': os.getenv('embedding_url', ''),
    })
    data['llm'].update({
        'provider': 'qwen',
        'model': os.getenv('llm_model', ''),
        'api_key': os.getenv('llm_api_key', ''),
        'base_url': os.getenv('llm_url', ''),
    })
    data['rerank'].update({
        'provider': 'qwen',
        'model': os.getenv('rerank_model', ''),
        'api_key': os.getenv('rerank_api_key', ''),
        'base_url': os.getenv('rerank_url', ''),
    })
    return data


def _dispose_engine(target) -> None:
    '''释放同步引擎连接池：None 与无 dispose 方法的对象直接跳过。'''
    dispose = getattr(target, 'dispose', None)
    if dispose is not None:
        dispose()


def _dispose_sync_handler(handler) -> None:
    '''释放门面持有的关系库与向量库连接池。'''
    store = getattr(getattr(handler, 'vector', None), 'store', None)
    _dispose_engine(getattr(store, 'engine', None))
    _dispose_engine(getattr(getattr(handler, 'db', None), 'engine', None))


async def _dispose_async_handler(handler) -> None:
    '''异步释放门面持有的关系库与向量库连接池。'''
    store = getattr(getattr(handler, 'vector', None), 'store', None)
    engine = getattr(store, 'engine', None)
    if engine is not None:
        await engine.dispose()
    db_engine = getattr(getattr(handler, 'db', None), 'engine', None)
    if db_engine is not None:
        await db_engine.dispose()

# 门面测试的后端参数矩阵：同步与异步各自独立，用例按后端组合参数化。
DB_BACKENDS = ["sqlite", "mysql", "pg"] # "sqlite", "mysql", "pg"
VECTOR_BACKENDS = ['milvus', 'pg']    # 'milvus', 'pg'


# 集成测试专用表前缀与集合名：与生产表隔离，避免干扰已有数据。
SYNC_TABLE_PREFIX = 'memories_facade_sync_'
SYNC_COLLECTION = 'memories_facade_sync'

@pytest.fixture(scope='module', params=DB_BACKENDS)
def db_backend(request):
    return request.param


@pytest.fixture(scope='module', params=VECTOR_BACKENDS)
def vector_backend(request):
    return request.param


class TestMemoriesIntegration:
    '''Memories 门面结合真实关系库与向量库的端到端集成测试。'''

    def test_01_real_connectivity(self, real_sync_handler):
        '''验证门面装配出的三层实现均为同步族，且关系库连接可用。'''
        handler = real_sync_handler
        assert _db_is_async(handler.db) is False
        assert _engine_is_async(handler.vector) is False
        assert _store_is_async(handler.vector.store) is False
        assert handler.db.engine is not None
        with handler.db.engine.connect() as conn:
            assert conn.execute(text('SELECT 1')).scalar() == 1


    def test_02_real_add_and_search(self, real_sync_handler):
        '''同一租户可检索到刚写入的记忆；换随机租户检索必定为空。'''
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()
        logger.info('theme:%s tenant_id:%s user_id:%s', theme, tenant_id, user_id)

        add_res = real_sync_handler.add(messages, tenant_id, user_id)
        assert isinstance(add_res, AddResult)
        assert len(add_res.results) >= 1

        query = random.choice(keywords)
        search_res = real_sync_handler.search(
            query, tenant_id, user_id, top_k=5, threshold=0.1, explain=True,
        )
        assert isinstance(search_res, SearchResult)
        assert len(search_res.results) > 0

        other_tenant_id, _ = random_identity()
        isolated = real_sync_handler.search(query, other_tenant_id, top_k=5, threshold=0.1)
        assert isinstance(isolated, SearchResult)
        assert isolated.results == []


    def test_03_real_get_and_page_query(self, real_sync_handler):
        '''按 id 读回单条，并验证分页查询覆盖到同一条记忆。'''
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()
        add_res = real_sync_handler.add(messages, tenant_id, user_id)
        assert add_res.results, '未抽取到事实，无法验证读取与分页'
        memory_id = add_res.results[0].id

        got = real_sync_handler.get(memory_id, tenant_id, user_id)
        assert isinstance(got, GetResult)
        assert got.result is not None
        assert got.result.id == memory_id
        assert got.result.tenant_id == tenant_id
        assert got.result.memory

        page = real_sync_handler.page_query(tenant_id, user_id, 0, 10)
        assert isinstance(page, PageQueryResult)
        assert page.total_size >= 1
        assert any(item.id == memory_id for item in page.results)


    def test_04_real_update_and_delete(self, real_sync_handler):
        '''更新正文后读回新值，删除后再读应为空。'''
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()
        add_res = real_sync_handler.add(messages, tenant_id, user_id)
        assert add_res.results, '未抽取到事实，无法验证更新与删除'
        memory_id = add_res.results[0].id

        new_text = random.choice(keywords) + ' 已更新为持久记忆'
        update_res = real_sync_handler.update(memory_id, tenant_id, user_id, text=new_text)
        assert isinstance(update_res, UpdateResult)
        assert update_res.result is not None
        assert update_res.result.memory == new_text

        delete_res = real_sync_handler.delete(memory_id, tenant_id, user_id)
        assert isinstance(delete_res, DeleteResult)
        assert delete_res.id == memory_id
        assert real_sync_handler.get(memory_id, tenant_id, user_id).result is None


    def test_05_real_load_from_json_text(self, db_backend, vector_backend):
        '''load 走 JSON 文本装配：只做构造不启动，因此无需释放连接池。'''
        data = _settings_data(db_backend, vector_backend, SYNC_TABLE_PREFIX, SYNC_COLLECTION)
        handler = Memories.load(json_text=json.dumps(data, ensure_ascii=False), use_env=False)
        assert isinstance(handler, MemoryHandler)
        assert handler.db.backend == db_backend
        assert handler.vector.backend == vector_backend
        assert handler.db.engine is None


DB_ASYNC_BACKENDS = ['sqlite_async', 'mysql_async', 'pg_async'] # 'sqlite_async', 'mysql_async', 'pg_async'
VECTOR_ASYNC_BACKENDS = ['milvus_async', 'pg_async'] # 'milvus_async', 'pg_async'

ASYNC_TABLE_PREFIX = 'memories_facade_async_'
ASYNC_COLLECTION = 'memories_facade_async'

@pytest.fixture(scope='module', params=DB_ASYNC_BACKENDS)
def db_async_backend(request):
    return request.param


@pytest.fixture(scope='module', params=VECTOR_ASYNC_BACKENDS)
def vector_async_backend(request):
    return request.param


@pytest.fixture(scope='module')
def real_sync_handler(db_backend, vector_backend):
    '''按后端矩阵装配 Memories 门面并启动，用例结束后释放连接池。'''
    settings = MemorySettings.model_validate(
        _settings_data(db_backend, vector_backend, SYNC_TABLE_PREFIX, SYNC_COLLECTION)
    )
    handler = Memories.create(settings)
    try:
        handler.start()
    except Exception as exc:
        pytest.skip('启动 Memories 失败：' + str(exc))
    yield handler
    _dispose_sync_handler(handler)

@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def real_async_handler(db_async_backend, vector_async_backend):
    '''按异步后端矩阵装配 AsyncMemories 门面并启动，用例结束后释放连接池。'''
    settings = MemorySettings.model_validate(
        _settings_data(
            db_async_backend, vector_async_backend, ASYNC_TABLE_PREFIX, ASYNC_COLLECTION,
        )
    )
    handler = AsyncMemories.create(settings)
    try:
        await handler.start()
    except Exception as exc:
        pytest.skip('启动 AsyncMemories 失败：' + str(exc))
    yield handler
    await _dispose_async_handler(handler)


@pytest.mark.asyncio(loop_scope='module')
class TestAsyncMemoriesIntegration:
    '''AsyncMemories 门面结合真实异步关系库与向量库的端到端集成测试。'''

    async def test_01_real_async_connectivity(self, real_async_handler):
        '''验证三层实现均为异步族，且异步关系库连接可用。'''
        handler = real_async_handler
        assert _db_is_async(handler.db) is True
        assert _engine_is_async(handler.vector) is True
        assert _store_is_async(handler.vector.store) is True
        assert handler.db.engine is not None
        async with handler.db.engine.connect() as conn:
            assert (await conn.execute(text('SELECT 1'))).scalar() == 1


    async def test_02_real_async_add_and_search(self, real_async_handler):
        '''异步写入后可检索；换随机租户检索必定为空。'''
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()
        logger.info('theme:%s tenant_id:%s user_id:%s', theme, tenant_id, user_id)

        add_res = await real_async_handler.add(messages, tenant_id, user_id)
        assert isinstance(add_res, AddResult)
        assert len(add_res.results) >= 1

        query = random.choice(keywords)
        search_res = await real_async_handler.search(
            query, tenant_id, user_id, top_k=5, threshold=0.1, explain=True,
        )
        assert isinstance(search_res, SearchResult)
        assert len(search_res.results) > 0

        other_tenant_id, _ = random_identity()
        isolated = await real_async_handler.search(query, other_tenant_id, top_k=5, threshold=0.1)
        assert isinstance(isolated, SearchResult)
        assert isolated.results == []


    async def test_03_real_async_update_and_delete(self, real_async_handler):
        '''异步更新后读回新值，删除后再读应为空。'''
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()
        add_res = await real_async_handler.add(messages, tenant_id, user_id)
        assert add_res.results, '未抽取到事实，无法验证更新与删除'
        memory_id = add_res.results[0].id

        new_text = random.choice(keywords) + ' 已更新为持久记忆'
        update_res = await real_async_handler.update(memory_id, tenant_id, user_id, text=new_text)
        assert isinstance(update_res, UpdateResult)
        assert update_res.result is not None
        assert update_res.result.memory == new_text

        delete_res = await real_async_handler.delete(memory_id, tenant_id, user_id)
        assert isinstance(delete_res, DeleteResult)
        assert delete_res.id == memory_id
        got = await real_async_handler.get(memory_id, tenant_id, user_id)
        assert got.result is None


    async def test_04_real_async_load_from_json_text(self, db_async_backend, vector_async_backend):
        '''异步 load 走 JSON 文本装配：只做构造不启动，因此无需释放连接池。'''
        data = _settings_data(
            db_async_backend, vector_async_backend, ASYNC_TABLE_PREFIX, ASYNC_COLLECTION,
        )
        handler = AsyncMemories.load(json_text=json.dumps(data, ensure_ascii=False), use_env=False)
        assert isinstance(handler, AsyncMemoryHandler)
        assert handler.db.backend == db_async_backend
        assert handler.vector.backend == vector_async_backend
        assert handler.db.engine is None
