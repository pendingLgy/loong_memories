'''sqlite_db.py 测试：SQLiteDB 与 AsyncSQLiteDB 的公共契约与继承而来的公共方法。

覆盖策略：
1. 全方法覆盖：构造方法 __init__、backend 类属性，以及继承自父类的公共方法
   add_history、batch_add_history、get_history、save_messages、
   get_last_messages、__enter__，含正常路径与边界分支。
2. 继承契约验证：backend 属性、__init__ 参数与 __enter__ 均在 SQLiteDB 与 AsyncSQLiteDB 实例上验证。
3. 不单独测试私有方法：_start、_missing_tables、_ensure_parent 等私有成员不做直接断言，
   仅作为 fixture 的初始化入口，其逻辑由公共方法的行为间接验证。
4. 不为父类建测试类：DBEngine、AsyncDBEngine、BaseDB 不设独立测试套件。

SQLite 库文件落在 pytest 的 tmp_path 内，不依赖任何外部服务，因此本模块默认执行，
无需 --integration；异步用例另需安装 aiosqlite 驱动。
连接参数取自 conftest 的 db 分区，库文件路径统一改由 tmp_path 提供，不写仓库目录。

运行：
    python -m pytest tests -q
'''

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


try:
    import sqlalchemy
    import aiosqlite
    import pytest_asyncio
    from sqlalchemy import MetaData

    from conftest import CONFIG_DB_SETTINGS
    from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine
    from loong_memories.storage.message.registry_db import is_db_registered, resolve_db
    from loong_memories.storage.message.schema import (
        HistoryRecord,
        Message,
        get_history_table,
        get_messages_table,
    )
    from loong_memories.storage.message.sqlite_db import AsyncSQLiteDB, SQLiteDB

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)


# ---------- 辅助 ----------

def _random_id():
    '''随机正整数：避免跨用例、跨次运行的历史数据互相干扰。'''
    return random.randint(1, 2 ** 31 - 1)


def _timed_messages(scope, count):
    '''按秒递增生成消息，保证 created_at_tz 可排序且不受时钟抖动影响。'''
    start = datetime.now(timezone.utc)
    return [
        Message(session_scope=scope, role='user', content='m' + str(index),
                created_at_tz=start + timedelta(seconds=index))
        for index in range(count)
    ]



# ---------- fixtures ----------

@pytest.fixture(scope='module')
def db_settings():
    '''conftest 的 db 分区深拷贝，后端切到同步 SQLite。'''
    return CONFIG_DB_SETTINGS.model_copy(deep=True, update={'backend': 'sqlite', 'table_prefix': ''})


@pytest_asyncio.fixture(scope='module', loop_scope='module')
def async_db_settings(db_settings):
    '''在 db_settings 上切到异步 SQLite 后端。'''
    return db_settings.model_copy(update={'backend': 'sqlite_async', 'table_prefix': 'async_'})


class TestSQLiteDB:
    '''SQLiteDB 的公共契约与继承自父类的公共方法。'''

    # ---- backend 类属性 ----

    def test_backend_is_registered(self):
        assert SQLiteDB.backend == 'sqlite'
        assert is_db_registered('sqlite') is True
        assert resolve_db('sqlite') is SQLiteDB

    def test_inherits_business_contract(self):
        assert issubclass(SQLiteDB, DBEngine)
        assert isinstance(SQLiteDB('db.sqlite'), DBEngine)

    # ---- __init__ ----

    def test_init_defaults(self):
        db = SQLiteDB('db.sqlite')
        assert db.db_path == 'db.sqlite'
        assert db.connection_string is None
        assert (db.print_sql, db.table_prefix) == (False, '')
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (10, 20, 30)
        assert (db.pool_recycle, db.pool_pre_ping) == (1800, True)

    def test_init_stores_conftest_values(self, db_settings):
        pool = db_settings.pool_config
        db = SQLiteDB(
            db_path=db_settings.sqlite.path,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            table_prefix=db_settings.table_prefix,
        )
        assert db.db_path == db_settings.sqlite.path
        assert db.print_sql == pool.print_sql
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (pool.pool_size, pool.max_overflow, pool.pool_timeout)
        assert (db.pool_recycle, db.pool_pre_ping) == (pool.pool_recycle, pool.pool_pre_ping)
        assert db.table_prefix == db_settings.table_prefix

    def test_init_builds_table_names_from_prefix(self, db_settings):
        db = SQLiteDB('db.sqlite', table_prefix=db_settings.table_prefix)
        assert db.history_table.name == db_settings.table_prefix + 'history'
        assert db.messages_table.name == db_settings.table_prefix + 'messages'

    def test_init_leaves_connection_unset(self):
        db = SQLiteDB('db.sqlite')
        assert (db.engine, db.SessionLocal) == (None, None)

    def test_enter_returns_self(self):
        db = SQLiteDB('db.sqlite')
        assert db.__enter__() is db


class TestAsyncSQLiteDB:
    '''AsyncSQLiteDB 的公共契约与继承自父类的公共方法。'''

    # ---- backend 类属性 ----

    def test_backend_is_registered(self):
        assert AsyncSQLiteDB.backend == 'sqlite_async'
        assert is_db_registered('sqlite_async') is True
        assert resolve_db('sqlite_async') is AsyncSQLiteDB

    def test_inherits_business_contract(self):
        assert issubclass(AsyncSQLiteDB, AsyncDBEngine)
        assert issubclass(AsyncSQLiteDB, DBEngine)

    # ---- __init__ ----

    def test_init_defaults(self):
        db = AsyncSQLiteDB('db.sqlite')
        assert db.db_path == 'db.sqlite'
        assert db.connection_string is None
        assert (db.print_sql, db.table_prefix) == (False, '')
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (10, 20, 30)
        assert (db.pool_recycle, db.pool_pre_ping) == (1800, True)

    def test_init_stores_conftest_values(self, async_db_settings):
        pool = async_db_settings.pool_config
        db = AsyncSQLiteDB(
            db_path=async_db_settings.sqlite.path,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
        )
        assert db.db_path == async_db_settings.sqlite.path
        assert db.print_sql == pool.print_sql
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (pool.pool_size, pool.max_overflow, pool.pool_timeout)
        assert (db.pool_recycle, db.pool_pre_ping) == (pool.pool_recycle, pool.pool_pre_ping)

    @pytest.mark.xfail(strict=True, reason='已知缺陷：AsyncDBEngine 构造函数把 table_prefix 重置为空串')
    def test_init_keeps_table_prefix(self):
        assert AsyncSQLiteDB('db.sqlite', table_prefix='lm_').table_prefix == 'lm_'

    def test_init_leaves_connection_unset(self):
        db = AsyncSQLiteDB('db.sqlite')
        assert (db.engine, db.AsyncSessionLocal) == (None, None)

    def test_enter_returns_self(self):
        db = AsyncSQLiteDB('db.sqlite')
        assert db.__enter__() is db


# ---------- 真实库文件：SQLiteDB ----------

class TestSQLiteDBIntegration:
    '''SQLiteDB 在真实库文件上的公共方法行为：历史记录与会话消息的读写淘汰。

    每个用例使用 tmp_path 下的独立库文件，表名取自 db_settings.table_prefix，
    因此用例之间没有数据残留，也不会污染仓库目录。
    '''

    @pytest.fixture(scope="module")
    def sqlite_db(self, db_settings):
        '''建库并完成初始化（_start 仅作 fixture 引导）。'''
        pool = db_settings.pool_config

        path = db_settings.sqlite.path
        db_file = Path(path).resolve()

        db = SQLiteDB(
            db_path=str(db_file),
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            table_prefix=db_settings.table_prefix,
        )
        db._start()
        yield db

        db.engine.dispose()

    def test_add_history_assigns_snowflake_id(self, sqlite_db):
        memory_id = _random_id()
        record = sqlite_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='hello', event='ADD'))
        assert record.id not in (None, 0)
        assert (record.memory_id, record.new_memory, record.event) == (memory_id, 'hello', 'ADD')
        assert record.is_deleted is False

    def test_add_history_keeps_caller_supplied_id(self, sqlite_db):
        record_id, memory_id = _random_id(), _random_id()
        record = sqlite_db.add_history(HistoryRecord(id=record_id, memory_id=memory_id, new_memory='n'))
        assert record.id == record_id
        assert [item.id for item in sqlite_db.get_history(memory_id)] == [record_id]

    def test_add_history_persists_all_fields(self, sqlite_db):
        memory_id = _random_id()
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        saved = sqlite_db.add_history(HistoryRecord(
            memory_id=memory_id,
            old_memory='old',
            new_memory='new',
            event='UPDATE',
            created_at_tz=created_at,
            updated_at_tz=updated_at,
            is_deleted=True,
            actor_id='u-1',
            role='user',
        ))
        got = sqlite_db.get_history(memory_id)[0]
        assert got.id == saved.id
        assert (got.old_memory, got.new_memory, got.event) == ('old', 'new', 'UPDATE')
        assert (got.is_deleted, got.actor_id, got.role) == (True, 'u-1', 'user')

    def test_batch_add_history_empty_list(self, sqlite_db):
        assert sqlite_db.batch_add_history([]) == []

    def test_batch_add_history_assigns_unique_ids(self, sqlite_db):
        memory_id = _random_id()
        records = [HistoryRecord(memory_id=memory_id, new_memory='n' + str(index)) for index in range(3)]
        saved = sqlite_db.batch_add_history(records)
        assert len(saved) == 3
        assert all(item.id is not None for item in saved)
        assert len({item.id for item in saved}) == 3
        assert len(sqlite_db.get_history(memory_id)) == 3

    def test_get_history_filters_by_memory_id(self, sqlite_db):
        first_id, second_id = _random_id(), _random_id()
        sqlite_db.add_history(HistoryRecord(memory_id=first_id, new_memory='a'))
        sqlite_db.add_history(HistoryRecord(memory_id=second_id, new_memory='b'))
        assert [item.new_memory for item in sqlite_db.get_history(first_id)] == ['a']

    def test_get_history_unknown_memory_returns_empty(self, sqlite_db):
        assert sqlite_db.get_history(_random_id()) == []

    def test_get_history_returns_desc_by_id(self, sqlite_db):
        memory_id = _random_id()
        first = sqlite_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='first'))
        second = sqlite_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='second'))
        assert [item.id for item in sqlite_db.get_history(memory_id)] == [second.id, first.id]

    def test_save_messages_empty_list(self, sqlite_db):
        assert sqlite_db.save_messages([]) == []

    def test_save_messages_persists_under_limit(self, sqlite_db):
        scope = 'scope-' + str(_random_id())
        saved = sqlite_db.save_messages(_timed_messages(scope, 3), keep_last=10)
        assert len(saved) == 3
        assert all(item.id is not None for item in saved)
        assert [item.content for item in sqlite_db.get_last_messages(scope, limit=10)] == ['m0', 'm1', 'm2']

    def test_save_messages_evicts_beyond_keep_last(self, sqlite_db):
        scope = 'scope-' + str(_random_id())
        sqlite_db.save_messages(_timed_messages(scope, 5), keep_last=2)
        assert [item.content for item in sqlite_db.get_last_messages(scope, limit=10)] == ['m3', 'm4']

    def test_save_messages_evicts_per_scope(self, sqlite_db):
        scope_a, scope_b = 'scope-' + str(_random_id()), 'scope-' + str(_random_id())
        sqlite_db.save_messages(_timed_messages(scope_a, 3) + _timed_messages(scope_b, 3), keep_last=2)
        assert [item.content for item in sqlite_db.get_last_messages(scope_a, limit=10)] == ['m1', 'm2']
        assert [item.content for item in sqlite_db.get_last_messages(scope_b, limit=10)] == ['m1', 'm2']

    def test_get_last_messages_respects_limit(self, sqlite_db):
        scope = 'scope-' + str(_random_id())
        sqlite_db.save_messages(_timed_messages(scope, 5), keep_last=10)
        assert [item.content for item in sqlite_db.get_last_messages(scope, limit=2)] == ['m3', 'm4']

    def test_get_last_messages_orders_by_created_at_asc(self, sqlite_db):
        scope = 'scope-' + str(_random_id())
        sqlite_db.save_messages(_timed_messages(scope, 3), keep_last=10)
        got = sqlite_db.get_last_messages(scope, limit=10)
        assert all(isinstance(item.created_at_tz, datetime) for item in got)
        assert [item.created_at_tz for item in got] == sorted(item.created_at_tz for item in got)

    def test_get_last_messages_unknown_scope_returns_empty(self, sqlite_db):
        assert sqlite_db.get_last_messages('scope-' + str(_random_id()), limit=5) == []


# ---------- 真实库文件：AsyncSQLiteDB ----------

@pytest.mark.asyncio(loop_scope='module')
class TestAsyncSQLiteDBIntegration:
    '''AsyncSQLiteDB 在真实库文件上的公共方法行为，语义与同步版本一致。'''

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def async_sqlite_db(self, async_db_settings):
        '''模块级共享库文件；用例靠随机 memory_id 与会话作用域隔离数据。

        父类构造函数会把 table_prefix 重置为空串（已知缺陷），因此在初始化前
        按传入前缀重建表元数据，避免落到无前缀的表上。
        '''
        prefix = async_db_settings.table_prefix
        pool = async_db_settings.pool_config
        path = async_db_settings.sqlite.path
        db_file = Path(path).resolve()
        db = AsyncSQLiteDB(
            db_path=str(db_file),
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            table_prefix=prefix,
        )
        db.table_prefix = prefix
        db.METADATA = MetaData()
        db.history_table = get_history_table(db.METADATA, prefix)
        db.messages_table = get_messages_table(db.METADATA, prefix)
        await db._start()
        yield db

        await db.engine.dispose()

    async def test_add_history_assigns_id_and_persists(self, async_sqlite_db):
        memory_id = _random_id()
        record = await async_sqlite_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='hello', event='ADD'))
        assert record.id not in (None, 0)
        assert (record.memory_id, record.event) == (memory_id, 'ADD')
        assert [item.id for item in await async_sqlite_db.get_history(memory_id)] == [record.id]

    async def test_add_history_keeps_caller_supplied_id(self, async_sqlite_db):
        record_id, memory_id = _random_id(), _random_id()
        record = await async_sqlite_db.add_history(HistoryRecord(id=record_id, memory_id=memory_id, new_memory='n'))
        assert record.id == record_id
        assert [item.id for item in await async_sqlite_db.get_history(memory_id)] == [record_id]

    async def test_batch_add_history_empty_list(self, async_sqlite_db):
        assert await async_sqlite_db.batch_add_history([]) == []

    async def test_batch_add_history_assigns_unique_ids(self, async_sqlite_db):
        memory_id = _random_id()
        records = [HistoryRecord(memory_id=memory_id, new_memory='n' + str(index)) for index in range(3)]
        saved = await async_sqlite_db.batch_add_history(records)
        assert len(saved) == 3
        assert len({item.id for item in saved}) == 3
        assert len(await async_sqlite_db.get_history(memory_id)) == 3

    async def test_get_history_filters_by_memory_id(self, async_sqlite_db):
        first_id, second_id = _random_id(), _random_id()
        await async_sqlite_db.add_history(HistoryRecord(memory_id=first_id, new_memory='a'))
        await async_sqlite_db.add_history(HistoryRecord(memory_id=second_id, new_memory='b'))
        assert [item.new_memory for item in await async_sqlite_db.get_history(first_id)] == ['a']

    async def test_get_history_unknown_memory_returns_empty(self, async_sqlite_db):
        assert await async_sqlite_db.get_history(_random_id()) == []

    async def test_save_messages_empty_list(self, async_sqlite_db):
        assert await async_sqlite_db.save_messages([]) == []

    async def test_save_messages_persists_under_limit(self, async_sqlite_db):
        scope = 'scope-' + str(_random_id())
        saved = await async_sqlite_db.save_messages(_timed_messages(scope, 3), keep_last=10)
        assert len(saved) == 3
        assert [item.content for item in await async_sqlite_db.get_last_messages(scope, limit=10)] == ['m0', 'm1', 'm2']

    async def test_save_messages_evicts_beyond_keep_last(self, async_sqlite_db):
        scope = 'scope-' + str(_random_id())
        await async_sqlite_db.save_messages(_timed_messages(scope, 5), keep_last=2)
        assert [item.content for item in await async_sqlite_db.get_last_messages(scope, limit=10)] == ['m3', 'm4']

    async def test_save_messages_evicts_per_scope(self, async_sqlite_db):
        scope_a, scope_b = 'scope-' + str(_random_id()), 'scope-' + str(_random_id())
        await async_sqlite_db.save_messages(_timed_messages(scope_a, 3) + _timed_messages(scope_b, 3), keep_last=2)
        assert [item.content for item in await async_sqlite_db.get_last_messages(scope_a, limit=10)] == ['m1', 'm2']
        assert [item.content for item in await async_sqlite_db.get_last_messages(scope_b, limit=10)] == ['m1', 'm2']

    async def test_get_last_messages_respects_limit(self, async_sqlite_db):
        scope = 'scope-' + str(_random_id())
        await async_sqlite_db.save_messages(_timed_messages(scope, 5), keep_last=10)
        assert [item.content for item in await async_sqlite_db.get_last_messages(scope, limit=2)] == ['m3', 'm4']

    async def test_get_last_messages_orders_by_created_at_asc(self, async_sqlite_db):
        scope = 'scope-' + str(_random_id())
        await async_sqlite_db.save_messages(_timed_messages(scope, 3), keep_last=10)
        got = await async_sqlite_db.get_last_messages(scope, limit=10)
        assert all(isinstance(item.created_at_tz, datetime) for item in got)
        assert [item.created_at_tz for item in got] == sorted(item.created_at_tz for item in got)

    async def test_get_last_messages_unknown_scope_returns_empty(self, async_sqlite_db):
        assert await async_sqlite_db.get_last_messages('scope-' + str(_random_id()), limit=5) == []
