'''mysql_db.py 测试包（合并版）：公共方法、私有方法与父类公共方法。

覆盖范围：
1. MySQLDB 与 AsyncMySQLDB 的构造、类属性、注册表登记
2. MySQLDB._start 与 AsyncMySQLDB._start（驱动校验、URL 归一化、引擎装配）
3. MySQLDB._missing_tables 与异步版本（INFORMATION_SCHEMA 比对）
4. db_engine.build_database 的 MySQL 分派
5. 父类 DBEngine 与 AsyncDBEngine 的公共业务方法（用 SQLite 引擎验证）
6. 真实 MySQL 集成用例（标记 integration，需 LOONG_TEST_MYSQL_DSN）

单元用例不连接真实 MySQL：驱动探测结果、SQLAlchemy 工厂函数与引擎均替换为测试替身。
父类用例用 SQLite 引擎验证与方言无关的通用语义。

本仓库 addopts 为 -m 'integration'，运行前需覆盖：
    python -m pytest tests -o addopts= -q
'''

from __future__ import annotations

import importlib.util
import inspect
import types
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip('sqlalchemy')

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from loong_memories.core.config_db import DBPoolSettings, DBSettings, MySQLSettings
from loong_memories.core.exceptions import ConfigError, OptionalDependencyError, StorageError
from loong_memories.storage.message import mysql_db
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine, build_database
from loong_memories.storage.message.mysql_db import AsyncMySQLDB, MySQLDB
from loong_memories.storage.message.registry_db import is_db_registered, resolve_db
from loong_memories.storage.message.schema import (
    HistoryRecord,
    Message,
    get_history_table,
    get_messages_table,
)

# ---------- 常量 ----------

SCHEME_SEP = chr(47) * 2

DSN = 'mysql' + SCHEME_SEP + 'svc:secret@db.example:3306/mem'

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)

BUSINESS_METHODS = ['add_history', 'batch_add_history', 'get_history', 'save_messages', 'get_last_messages']

HAS_ASYNCIO_PLUGIN = importlib.util.find_spec('pytest_asyncio') is not None
needs_asyncio = pytest.mark.skipif(not HAS_ASYNCIO_PLUGIN, reason='异步用例需要 pytest-asyncio 插件，且 asyncio_mode 为 auto')

# ---------- 测试替身：引擎与会话 ----------

class _FakeResult:
    '''最小结果对象：只提供 fetchall。'''

    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _SyncConn:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def execute(self, statement, params=None):
        self._calls.append((statement, dict(params or {})))
        return _FakeResult(self._rows)


class _SyncBegin:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc_info):
        return False


class _AsyncConn:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    async def execute(self, statement, params=None):
        self._calls.append((statement, dict(params or {})))
        return _FakeResult(self._rows)


class _AsyncBegin:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


def _sync_engine(rows, calls):
    '''构造可记录执行的同步引擎替身。'''
    conn = _SyncConn(rows, calls)
    return types.SimpleNamespace(begin=lambda: _SyncBegin(conn))


def _async_engine(rows, calls):
    '''构造可记录执行的异步引擎替身。'''
    conn = _AsyncConn(rows, calls)
    return types.SimpleNamespace(begin=lambda: _AsyncBegin(conn))

# ---------- 测试替身：最小的数据库子类 ----------

class _SyncTestDB(DBEngine):
    '''最小同步子类：SQLite 引擎配合父类通用实现。backend 为空，避免污染全局注册表。'''

    backend = None

    def __init__(self, db_path, table_prefix=''):
        self.db_path = str(db_path)
        super().__init__(table_prefix=table_prefix)

    def _start(self):
        self.engine = create_engine(URL.create('sqlite', database=self.db_path))
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_db(self.engine, self.SessionLocal)
        return self.engine

    def _missing_tables(self):
        all_tables = list(self.METADATA.sorted_tables)
        if not all_tables:
            return []
        with self.engine.connect() as conn:
            result = conn.execute(text('SELECT name FROM sqlite_master WHERE type = :t'), {'t': 'table'})
            existing = {row[0] for row in result.fetchall()}
        return [table for table in all_tables if table.name not in existing]


class _AsyncTestDB(AsyncDBEngine):
    '''最小异步子类：SQLite 异步引擎配合父类通用实现。'''

    backend = None

    def __init__(self, db_path, table_prefix=''):
        self.db_path = str(db_path)
        super().__init__(table_prefix=table_prefix)
        # 父类构造会重置 table_prefix 与 METADATA，这里按调用方传入值重建，隔离该已知缺陷。
        self.table_prefix = table_prefix
        self.METADATA = MetaData()
        self.history_table = get_history_table(self.METADATA, table_prefix)
        self.messages_table = get_messages_table(self.METADATA, table_prefix)

    async def _start(self):
        self.engine = create_async_engine(URL.create('sqlite+aiosqlite', database=self.db_path))
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        await self._init_db(self.engine, self.AsyncSessionLocal)
        return self.engine

    async def _missing_tables(self):
        all_tables = list(self.METADATA.sorted_tables)
        if not all_tables:
            return []
        async with self.engine.connect() as conn:
            result = await conn.execute(text('SELECT name FROM sqlite_master WHERE type = :t'), {'t': 'table'})
            existing = {row[0] for row in result.fetchall()}
        return [table for table in all_tables if table.name not in existing]

# ---------- fixtures：驱动与引擎打桩 ----------

@pytest.fixture
def sync_driver(monkeypatch):
    '''强制同步驱动为 pymysql，与真实安装状态解耦。'''
    monkeypatch.setattr(mysql_db, '_MYSQL_DRIVER', 'pymysql')
    monkeypatch.setattr(mysql_db, '_MYSQL_ERROR', None)
    return 'pymysql'

@pytest.fixture
def async_driver(monkeypatch):
    '''强制异步驱动为 aiomysql，与真实安装状态解耦。'''
    monkeypatch.setattr(mysql_db, '_ASYNC_MYSQL_DRIVER', 'aiomysql')
    monkeypatch.setattr(mysql_db, '_ASYNC_MYSQL_ERROR', None)
    return 'aiomysql'

@pytest.fixture
def no_sync_driver(monkeypatch):
    '''模拟同步驱动缺失。'''
    monkeypatch.setattr(mysql_db, '_MYSQL_DRIVER', '')
    monkeypatch.setattr(mysql_db, '_MYSQL_ERROR', ImportError('missing sync driver'))

@pytest.fixture
def no_async_driver(monkeypatch):
    '''模拟异步驱动缺失。'''
    monkeypatch.setattr(mysql_db, '_ASYNC_MYSQL_DRIVER', '')
    monkeypatch.setattr(mysql_db, '_ASYNC_MYSQL_ERROR', ImportError('missing async driver'))

@pytest.fixture
def sync_engine_capture(monkeypatch):
    '''接管 create_engine 与 sessionmaker，记录调用参数。'''
    box = {
        'engine_calls': [],
        'sessionmaker_calls': [],
        'engine': types.SimpleNamespace(kind='sync-engine'),
    }

    def fake_create_engine(url, **kwargs):
        box['engine_calls'].append((url, kwargs))
        return box['engine']

    def fake_sessionmaker(**kwargs):
        box['sessionmaker_calls'].append(kwargs)
        return types.SimpleNamespace(kind='sync-session')

    monkeypatch.setattr(mysql_db, 'create_engine', fake_create_engine)
    monkeypatch.setattr(mysql_db, 'sessionmaker', fake_sessionmaker)
    return box

@pytest.fixture
def async_engine_capture(monkeypatch):
    '''接管 create_async_engine 与 async_sessionmaker，可注入构造异常。'''
    box = {
        'engine_calls': [],
        'sessionmaker_calls': [],
        'engine': types.SimpleNamespace(kind='async-engine'),
        'error': None,
    }

    def fake_create_async_engine(url, **kwargs):
        box['engine_calls'].append((url, kwargs))
        if box['error'] is not None:
            raise box['error']
        return box['engine']

    def fake_async_sessionmaker(**kwargs):
        box['sessionmaker_calls'].append(kwargs)
        return types.SimpleNamespace(kind='async-session')

    monkeypatch.setattr(mysql_db, 'create_async_engine', fake_create_async_engine)
    monkeypatch.setattr(mysql_db, 'async_sessionmaker', fake_async_sessionmaker)
    return box

# ---------- 公共方法：MySQLDB ----------

class TestMySQLDBPublic:
    '''MySQLDB 公共契约：类属性、注册表登记、构造参数与父类继承关系。'''

    def test_backend_name(self):
        assert MySQLDB.backend == 'mysql'

    def test_registered_under_backend_name(self):
        assert is_db_registered('mysql') is True
        assert resolve_db('mysql') is MySQLDB

    def test_inherits_db_engine(self):
        assert issubclass(MySQLDB, DBEngine)
        assert isinstance(MySQLDB(), DBEngine)

    def test_engine_is_lazy(self):
        '''构造阶段不建立连接，引擎与会话工厂保持为空。'''
        db = MySQLDB()
        assert db.engine is None
        assert db.SessionLocal is None

    def test_table_prefix_flows_into_table_names(self):
        db = MySQLDB(table_prefix='lm_')
        assert db.history_table.name == 'lm_history'
        assert db.messages_table.name == 'lm_messages'

    def test_default_construction_values(self):
        db = MySQLDB()
        assert db.connection_string is None
        assert db.host == 'localhost'
        assert db.port == 3306
        assert db.user == 'root'
        assert db.password == ''
        assert db.database == ''
        assert db.charset == 'utf8mb4'
        assert db.print_sql is False
        assert db.pool_size == 10
        assert db.max_overflow == 20
        assert db.pool_timeout == 30
        assert db.pool_recycle == 1800
        assert db.pool_pre_ping is True
        assert db.table_prefix == ''

    def test_custom_construction_values(self):
        db = MySQLDB(
            connection_string=DSN,
            host='db.example',
            port=3307,
            user='svc',
            password='secret',
            database='mem',
            charset='utf8mb4',
            print_sql=True,
            pool_size=3,
            max_overflow=7,
            pool_timeout=11,
            pool_recycle=120,
            pool_pre_ping=False,
            table_prefix='lm_',
        )
        assert db.connection_string == DSN
        assert db.host == 'db.example'
        assert db.port == 3307
        assert db.user == 'svc'
        assert db.password == 'secret'
        assert db.database == 'mem'
        assert db.print_sql is True
        assert db.pool_size == 3
        assert db.max_overflow == 7
        assert db.pool_timeout == 11
        assert db.pool_recycle == 120
        assert db.pool_pre_ping is False
        assert db.table_prefix == 'lm_'

    @pytest.mark.parametrize('name', BUSINESS_METHODS)
    def test_business_methods_are_inherited_from_parent(self, name):
        '''mysql_db 不覆写业务方法，全部复用父类实现。'''
        assert getattr(MySQLDB, name) is getattr(DBEngine, name)


class TestAsyncMySQLDBPublic:
    '''AsyncMySQLDB 公共契约。'''

    def test_backend_name(self):
        assert AsyncMySQLDB.backend == 'mysql_async'

    def test_registered_under_backend_name(self):
        assert is_db_registered('mysql_async') is True
        assert resolve_db('mysql_async') is AsyncMySQLDB

    def test_inherits_async_db_engine(self):
        assert issubclass(AsyncMySQLDB, AsyncDBEngine)

    def test_engine_is_lazy(self):
        db = AsyncMySQLDB()
        assert db.engine is None
        assert db.AsyncSessionLocal is None

    @pytest.mark.parametrize('name', BUSINESS_METHODS)
    def test_business_methods_are_coroutine_functions(self, name):
        '''异步后端的业务方法必须是协程，由父类 AsyncDBEngine 重写。'''
        assert inspect.iscoroutinefunction(getattr(AsyncMySQLDB, name)) is True

    @pytest.mark.parametrize('name', ['_start', '_missing_tables'])
    def test_dialect_hooks_are_coroutine_functions(self, name):
        assert inspect.iscoroutinefunction(getattr(AsyncMySQLDB, name)) is True


# ---------- 公共方法：build_database 的 MySQL 分派 ----------

class TestBuildDatabaseMySQL:
    '''db_engine.build_database 对 MySQL 分区的装配行为。'''

    def test_build_sync_mysql(self):
        settings = DBSettings(
            backend='mysql',
            table_prefix='lm_',
            mysql=MySQLSettings(
                host='db.example',
                port=3307,
                user='svc',
                password='secret',
                database='mem',
                charset='utf8mb4',
            ),
            pool_config=DBPoolSettings(
                print_sql=True,
                pool_size=3,
                max_overflow=7,
                pool_timeout=11,
                pool_recycle=120,
                pool_pre_ping=True,
            ),
        )
        db = build_database(settings)
        assert isinstance(db, MySQLDB)
        assert db.host == 'db.example'
        assert db.port == 3307
        assert db.user == 'svc'
        assert db.database == 'mem'
        assert db.charset == 'utf8mb4'
        assert db.table_prefix == 'lm_'
        assert db.print_sql is True
        assert db.pool_size == 3
        assert db.max_overflow == 7
        assert db.pool_timeout == 11
        assert db.pool_recycle == 120
        assert db.pool_pre_ping is True
        assert db.history_table.name == 'lm_history'

    def test_build_async_mysql(self):
        settings = DBSettings(backend='mysql_async', mysql=MySQLSettings(host='db.example'))
        db = build_database(settings)
        assert isinstance(db, AsyncMySQLDB)
        assert db.host == 'db.example'

    @pytest.mark.xfail(strict=True, reason='已知缺陷：AsyncDBEngine 构造函数调用父类时把 table_prefix 重置为空串')
    def test_build_async_mysql_keeps_table_prefix(self):
        settings = DBSettings(backend='mysql_async', table_prefix='lm_')
        db = build_database(settings)
        assert db.table_prefix == 'lm_'
        assert db.history_table.name == 'lm_history'

    def test_unknown_backend_raises_config_error(self):
        '''配置层 Literal 已限制取值，这里用 model_construct 绕过校验以覆盖防御分支。'''
        settings = DBSettings.model_construct(
            backend='oracle',
            table_prefix='',
            pool_config=DBPoolSettings(),
        )
        with pytest.raises(ConfigError):
            build_database(settings)

# ---------- 私有方法：MySQLDB._start ----------

class TestMySQLDBStart:
    '''MySQLDB._start：驱动校验、URL 归一化、引擎与会话工厂装配。'''

    def test_raises_when_driver_missing(self, no_sync_driver):
        db = MySQLDB()
        with pytest.raises(OptionalDependencyError):
            db._start()

    def test_returns_engine_and_binds_sessionmaker(self, sync_driver, sync_engine_capture, monkeypatch):
        db = MySQLDB()
        monkeypatch.setattr(db, '_init_db', lambda engine, session: None)
        engine = db._start()
        assert engine is sync_engine_capture['engine']
        assert db.engine is sync_engine_capture['engine']
        assert db.SessionLocal.kind == 'sync-session'
        assert sync_engine_capture['sessionmaker_calls'][0]['bind'] is sync_engine_capture['engine']

    def test_builds_url_from_discrete_params(self, sync_driver, sync_engine_capture, monkeypatch):
        db = MySQLDB(
            host='db.example',
            port=3307,
            user='svc',
            password='secret',
            database='mem',
            charset='utf8mb4',
        )
        monkeypatch.setattr(db, '_init_db', lambda engine, session: None)
        db._start()
        url = sync_engine_capture['engine_calls'][0][0]
        assert url.startswith('mysql+pymysql' + SCHEME_SEP)
        assert 'db.example' in url
        assert ':3307' in url
        assert 'mem' in url
        assert 'charset=utf8mb4' in url
        assert db.connection_string == url

    def test_passes_pool_options_to_engine(self, sync_driver, sync_engine_capture, monkeypatch):
        db = MySQLDB(
            print_sql=True,
            pool_size=3,
            max_overflow=7,
            pool_timeout=11,
            pool_recycle=120,
            pool_pre_ping=False,
        )
        monkeypatch.setattr(db, '_init_db', lambda engine, session: None)
        db._start()
        kwargs = sync_engine_capture['engine_calls'][0][1]
        assert kwargs['echo'] is True
        assert kwargs['pool_size'] == 3
        assert kwargs['max_overflow'] == 7
        assert kwargs['pool_timeout'] == 11
        assert kwargs['pool_recycle'] == 120
        assert kwargs['pool_pre_ping'] is False

    def test_calls_init_db_with_engine_and_session(self, sync_driver, sync_engine_capture, monkeypatch):
        db = MySQLDB()
        calls = []
        monkeypatch.setattr(db, '_init_db', lambda engine, session: calls.append((engine, session)))
        db._start()
        assert len(calls) == 1
        assert calls[0][0] is sync_engine_capture['engine']
        assert calls[0][1].kind == 'sync-session'

    def test_normalizes_mysql_scheme(self, sync_driver, sync_engine_capture, monkeypatch):
        '''传入 mysql 双斜杠前缀时应改写为具体同步驱动。'''
        dsn = 'mysql' + SCHEME_SEP + 'svc:secret@db.example:3306'
        db = MySQLDB(connection_string=dsn)
        monkeypatch.setattr(db, '_init_db', lambda engine, session: None)
        db._start()
        expected = 'mysql+pymysql' + SCHEME_SEP + 'svc:secret@db.example:3306'
        assert sync_engine_capture['engine_calls'][0][0] == expected
        assert db.connection_string == expected

    def test_normalizes_mysqlconnector_scheme(self, sync_driver, sync_engine_capture, monkeypatch):
        '''传入 mysqlconnector 前缀时应改写为实际选中的驱动。'''
        dsn = 'mysql+mysqlconnector' + SCHEME_SEP + 'svc@db.example:3306'
        db = MySQLDB(connection_string=dsn)
        monkeypatch.setattr(db, '_init_db', lambda engine, session: None)
        db._start()
        expected = 'mysql+pymysql' + SCHEME_SEP + 'svc@db.example:3306'
        assert sync_engine_capture['engine_calls'][0][0] == expected

    def test_leaves_unknown_scheme_untouched(self, sync_driver, sync_engine_capture, monkeypatch):
        '''同步实现的白名单不含异步前缀，此处锁定当前直传行为。'''
        dsn = 'mysql+aiomysql' + SCHEME_SEP + 'svc@db.example:3306'
        db = MySQLDB(connection_string=dsn)
        monkeypatch.setattr(db, '_init_db', lambda engine, session: None)
        db._start()
        assert sync_engine_capture['engine_calls'][0][0] == dsn

# ---------- 私有方法：MySQLDB._missing_tables ----------

class TestMySQLDBMissingTables:
    '''MySQLDB._missing_tables：通过 INFORMATION_SCHEMA 比对已建表。'''

    def test_returns_absent_tables(self):
        db = MySQLDB()
        calls = []
        db.engine = _sync_engine([('history',)], calls)
        missing = db._missing_tables()
        assert [table.name for table in missing] == ['messages']

    def test_returns_empty_when_all_present(self):
        db = MySQLDB()
        calls = []
        db.engine = _sync_engine([('history',), ('messages',)], calls)
        assert db._missing_tables() == []

    def test_returns_all_when_database_empty(self):
        db = MySQLDB()
        calls = []
        db.engine = _sync_engine([], calls)
        missing = db._missing_tables()
        assert {table.name for table in missing} == {'history', 'messages'}

    def test_queries_information_schema_with_bound_names(self):
        db = MySQLDB()
        calls = []
        db.engine = _sync_engine([], calls)
        db._missing_tables()
        statement, params = calls[0]
        sql = str(statement)
        assert 'INFORMATION_SCHEMA.TABLES' in sql
        assert 'TABLE_SCHEMA = DATABASE()' in sql
        assert sorted(params) == ['t0', 't1']
        assert set(params.values()) == {'history', 'messages'}

    def test_applies_table_prefix(self):
        db = MySQLDB(table_prefix='lm_')
        calls = []
        db.engine = _sync_engine([], calls)
        db._missing_tables()
        assert set(calls[0][1].values()) == {'lm_history', 'lm_messages'}

    def test_empty_metadata_short_circuits_without_engine(self):
        '''无表时直接返回，不触碰引擎；此处 engine 故意保持为 None。'''
        db = MySQLDB()
        db.METADATA = MetaData()
        db.history_table = None
        db.messages_table = None
        assert db._missing_tables() == []
        assert db.engine is None

# ---------- 私有方法：AsyncMySQLDB._start 与 _missing_tables ----------

@needs_asyncio
class TestAsyncMySQLDBStart:
    '''AsyncMySQLDB._start：驱动校验、URL 归一化、异步引擎与会话工厂装配。'''

    async def test_raises_when_driver_missing(self, no_async_driver):
        db = AsyncMySQLDB()
        with pytest.raises(OptionalDependencyError):
            await db._start()

    async def test_returns_engine_and_session_factory(self, async_driver, async_engine_capture, monkeypatch):
        db = AsyncMySQLDB(host='db.example', port=3307, user='svc', password='secret', database='mem')
        calls = []

        async def fake_init_db(engine, session):
            calls.append((engine, session))

        monkeypatch.setattr(db, '_init_db', fake_init_db)
        engine = await db._start()
        url = async_engine_capture['engine_calls'][0][0]
        assert engine is async_engine_capture['engine']
        assert url.startswith('mysql+aiomysql' + SCHEME_SEP)
        assert 'db.example' in url
        assert ':3307' in url
        assert len(calls) == 1
        assert calls[0][0] is async_engine_capture['engine']

    async def test_session_factory_uses_expire_on_commit_false(self, async_driver, async_engine_capture, monkeypatch):
        db = AsyncMySQLDB()

        async def fake_init_db(engine, session):
            return None

        monkeypatch.setattr(db, '_init_db', fake_init_db)
        await db._start()
        kwargs = async_engine_capture['sessionmaker_calls'][0]
        assert kwargs['expire_on_commit'] is False
        assert kwargs['bind'] is async_engine_capture['engine']
        assert db.AsyncSessionLocal.kind == 'async-session'

    async def test_passes_pool_options_to_engine(self, async_driver, async_engine_capture, monkeypatch):
        db = AsyncMySQLDB(pool_size=4, max_overflow=6, pool_recycle=60, pool_pre_ping=True)

        async def fake_init_db(engine, session):
            return None

        monkeypatch.setattr(db, '_init_db', fake_init_db)
        await db._start()
        kwargs = async_engine_capture['engine_calls'][0][1]
        assert kwargs['pool_size'] == 4
        assert kwargs['max_overflow'] == 6
        assert kwargs['pool_recycle'] == 60
        assert kwargs['pool_pre_ping'] is True

    async def test_wraps_engine_error_as_storage_error(self, async_driver, async_engine_capture):
        db = AsyncMySQLDB()
        async_engine_capture['error'] = RuntimeError('boom')
        with pytest.raises(StorageError):
            await db._start()

    async def test_normalizes_mysql_scheme(self, async_driver, async_engine_capture, monkeypatch):
        dsn = 'mysql' + SCHEME_SEP + 'svc:secret@db.example:3306'
        db = AsyncMySQLDB(connection_string=dsn)

        async def fake_init_db(engine, session):
            return None

        monkeypatch.setattr(db, '_init_db', fake_init_db)
        await db._start()
        expected = 'mysql+aiomysql' + SCHEME_SEP + 'svc:secret@db.example:3306'
        assert async_engine_capture['engine_calls'][0][0] == expected


@needs_asyncio
class TestAsyncMySQLDBMissingTables:
    '''AsyncMySQLDB._missing_tables：异步引擎下的 INFORMATION_SCHEMA 比对。'''

    async def test_returns_absent_tables(self):
        db = AsyncMySQLDB()
        calls = []
        db.engine = _async_engine([('history',)], calls)
        missing = await db._missing_tables()
        assert [table.name for table in missing] == ['messages']

    async def test_returns_empty_when_all_present(self):
        db = AsyncMySQLDB()
        calls = []
        db.engine = _async_engine([('history',), ('messages',)], calls)
        assert await db._missing_tables() == []

    async def test_returns_all_when_database_empty(self):
        db = AsyncMySQLDB()
        calls = []
        db.engine = _async_engine([], calls)
        missing = await db._missing_tables()
        assert {table.name for table in missing} == {'history', 'messages'}

    async def test_queries_information_schema_with_bound_names(self):
        db = AsyncMySQLDB()
        calls = []
        db.engine = _async_engine([], calls)
        await db._missing_tables()
        statement, params = calls[0]
        sql = str(statement)
        assert 'INFORMATION_SCHEMA.TABLES' in sql
        assert set(params.values()) == {'history', 'messages'}

    async def test_applies_table_prefix(self):
        db = AsyncMySQLDB(table_prefix='lm_')
        calls = []
        db.engine = _async_engine([], calls)
        await db._missing_tables()
        assert set(calls[0][1].values()) == {'lm_history', 'lm_messages'}

    async def test_empty_metadata_short_circuits_without_engine(self):
        db = AsyncMySQLDB()
        db.METADATA = MetaData()
        db.history_table = None
        db.messages_table = None
        assert await db._missing_tables() == []
        assert db.engine is None

# ---------- 父类公共方法：DBEngine ----------

@pytest.fixture
def sync_db(tmp_path):
    '''构造已建表的同步 SQLite 测试实例。'''
    db = _SyncTestDB(tmp_path / 'engine.db')
    db._start()
    return db


class TestDBEnginePublic:
    '''DBEngine 公共方法：历史记录与会话消息的读写。'''

    def test_add_history_assigns_snowflake_id(self, sync_db):
        record = sync_db.add_history(HistoryRecord(memory_id=1, new_memory='n1', event='ADD'))
        assert record.id is not None
        assert record.id != 0
        assert record.memory_id == 1
        assert record.new_memory == 'n1'
        assert record.event == 'ADD'
        assert record.is_deleted is False

    def test_add_history_keeps_caller_supplied_id(self, sync_db):
        record = sync_db.add_history(HistoryRecord(id=42, memory_id=2, new_memory='n2'))
        assert record.id == 42
        assert [item.id for item in sync_db.get_history(2)] == [42]

    def test_batch_add_history_empty_list(self, sync_db):
        assert sync_db.batch_add_history([]) == []

    def test_batch_add_history_assigns_ids(self, sync_db):
        records = [HistoryRecord(memory_id=9, new_memory='n' + str(i)) for i in range(3)]
        saved = sync_db.batch_add_history(records)
        assert len(saved) == 3
        assert all(item.id is not None for item in saved)
        assert len({item.id for item in saved}) == 3
        assert len(sync_db.get_history(9)) == 3

    def test_get_history_unknown_memory_returns_empty(self, sync_db):
        assert sync_db.get_history(404) == []

    def test_get_history_filters_by_memory_id(self, sync_db):
        sync_db.add_history(HistoryRecord(memory_id=1, new_memory='a'))
        sync_db.add_history(HistoryRecord(memory_id=2, new_memory='b'))
        got = sync_db.get_history(1)
        assert len(got) == 1
        assert got[0].new_memory == 'a'

    def test_get_history_current_order_is_desc_by_id(self, sync_db):
        '''锁定当前实现：按 id 降序返回，与 docstring 表述存在差异。'''
        first = sync_db.add_history(HistoryRecord(memory_id=5, new_memory='first'))
        second = sync_db.add_history(HistoryRecord(memory_id=5, new_memory='second'))
        assert [item.id for item in sync_db.get_history(5)] == [second.id, first.id]

    def test_save_messages_empty_list(self, sync_db):
        assert sync_db.save_messages([]) == []

    def test_save_messages_persists_when_under_limit(self, sync_db):
        messages = [
            Message(session_scope='s1', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(3)
        ]
        saved = sync_db.save_messages(messages, keep_last=10)
        assert len(saved) == 3
        assert all(item.id is not None for item in saved)
        assert [item.content for item in sync_db.get_last_messages('s1', limit=10)] == ['m0', 'm1', 'm2']

    def test_save_messages_evicts_beyond_keep_last(self, sync_db):
        messages = [
            Message(session_scope='s1', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(5)
        ]
        sync_db.save_messages(messages, keep_last=2)
        assert [item.content for item in sync_db.get_last_messages('s1', limit=10)] == ['m3', 'm4']

    def test_save_messages_evicts_per_scope(self, sync_db):
        messages = []
        for scope in ('s1', 's2'):
            for i in range(3):
                messages.append(
                    Message(
                        session_scope=scope,
                        role='user',
                        content=scope + '-' + str(i),
                        created_at_tz=BASE_TIME + timedelta(seconds=i),
                    )
                )
        sync_db.save_messages(messages, keep_last=2)
        assert [item.content for item in sync_db.get_last_messages('s1', limit=10)] == ['s1-1', 's1-2']
        assert [item.content for item in sync_db.get_last_messages('s2', limit=10)] == ['s2-1', 's2-2']

    def test_get_last_messages_respects_limit(self, sync_db):
        messages = [
            Message(session_scope='s1', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(5)
        ]
        sync_db.save_messages(messages, keep_last=10)
        assert [item.content for item in sync_db.get_last_messages('s1', limit=2)] == ['m3', 'm4']

    def test_get_last_messages_unknown_scope_returns_empty(self, sync_db):
        assert sync_db.get_last_messages('nope') == []

    def test_enter_returns_self(self, sync_db):
        '''父类只实现了 __enter__，这里只验证其返回值。'''
        assert sync_db.__enter__() is sync_db

# ---------- 父类公共方法：AsyncDBEngine ----------

@needs_asyncio
class TestAsyncDBEnginePublic:
    '''AsyncDBEngine 公共方法：与同步版本语义一致。'''

    @pytest.fixture
    async def async_db(self, tmp_path):
        pytest.importorskip('aiosqlite')
        db = _AsyncTestDB(tmp_path / 'engine_async.db')
        await db._start()
        yield db
        await db.engine.dispose()

    async def test_add_history_assigns_id(self, async_db):
        record = await async_db.add_history(HistoryRecord(memory_id=1, new_memory='n1'))
        assert record.id is not None
        assert record.id != 0

    async def test_batch_add_history_empty_list(self, async_db):
        assert await async_db.batch_add_history([]) == []

    async def test_batch_add_history_assigns_ids(self, async_db):
        records = [HistoryRecord(memory_id=3, new_memory='n' + str(i)) for i in range(3)]
        saved = await async_db.batch_add_history(records)
        assert len({item.id for item in saved}) == 3
        assert len(await async_db.get_history(3)) == 3

    async def test_get_history_filters_by_memory_id(self, async_db):
        await async_db.add_history(HistoryRecord(memory_id=1, new_memory='a'))
        await async_db.add_history(HistoryRecord(memory_id=2, new_memory='b'))
        got = await async_db.get_history(1)
        assert [item.new_memory for item in got] == ['a']

    async def test_get_history_unknown_memory_returns_empty(self, async_db):
        assert await async_db.get_history(404) == []

    async def test_save_messages_evicts_beyond_keep_last(self, async_db):
        messages = [
            Message(session_scope='s1', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(5)
        ]
        await async_db.save_messages(messages, keep_last=2)
        got = await async_db.get_last_messages('s1', limit=10)
        assert [item.content for item in got] == ['m3', 'm4']

    async def test_get_last_messages_respects_limit(self, async_db):
        messages = [
            Message(session_scope='s1', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(5)
        ]
        await async_db.save_messages(messages, keep_last=10)
        got = await async_db.get_last_messages('s1', limit=2)
        assert [item.content for item in got] == ['m3', 'm4']

    async def test_get_last_messages_unknown_scope_returns_empty(self, async_db):
        assert await async_db.get_last_messages('nope') == []

    async def test_enter_returns_self(self, async_db):
        assert async_db.__enter__() is async_db

# ---------- 真实 MySQL 集成用例 ----------

@pytest.mark.integration
class TestMySQLDBIntegration:
    '''真实 MySQL 下的读写链路；需 LOONG_TEST_MYSQL_DSN，未设置时由 mysql_dsn 跳过。'''

    @pytest.fixture
    def mysql_db(self, mysql_dsn):
        '''建立使用独立表前缀的实例，用例结束后清理生成的表。'''
        prefix = 'it' + uuid.uuid4().hex[:8] + '_'
        db = MySQLDB(connection_string=mysql_dsn, table_prefix=prefix)
        db._start()
        yield db
        with db.engine.begin() as conn:
            conn.execute(text('DROP TABLE IF EXISTS `' + prefix + 'messages`'))
            conn.execute(text('DROP TABLE IF EXISTS `' + prefix + 'history`'))
        db.engine.dispose()

    def test_start_creates_all_tables(self, mysql_db):
        assert mysql_db._missing_tables() == []

    def test_add_and_get_history(self, mysql_db):
        saved = mysql_db.add_history(HistoryRecord(memory_id=101, new_memory='hello', event='ADD'))
        assert saved.id is not None
        assert [item.new_memory for item in mysql_db.get_history(101)] == ['hello']

    def test_batch_add_history(self, mysql_db):
        records = [HistoryRecord(memory_id=202, new_memory='n' + str(i)) for i in range(3)]
        saved = mysql_db.batch_add_history(records)
        assert len(saved) == 3
        assert len(mysql_db.get_history(202)) == 3

    def test_save_and_get_messages(self, mysql_db):
        messages = [
            Message(session_scope='s', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(3)
        ]
        mysql_db.save_messages(messages, keep_last=10)
        assert [item.content for item in mysql_db.get_last_messages('s', limit=10)] == ['m0', 'm1', 'm2']

    def test_evict_keeps_last(self, mysql_db):
        messages = [
            Message(session_scope='s', role='user', content='m' + str(i), created_at_tz=BASE_TIME + timedelta(seconds=i))
            for i in range(5)
        ]
        mysql_db.save_messages(messages, keep_last=2)
        assert [item.content for item in mysql_db.get_last_messages('s', limit=10)] == ['m3', 'm4']

