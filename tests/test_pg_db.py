'''pg_db.py 测试：PGDB 与 AsyncPGDB 的公共契约与继承而来的公共方法。

覆盖策略：
1. 全方法覆盖：构造方法 __init__、backend 类属性，以及继承自父类的公共方法
   add_history、batch_add_history、get_history、save_messages、
   get_last_messages、__enter__，含正常路径与边界分支。
2. 继承契约验证：backend 属性、__init__ 参数与 __enter__ 均在 PGDB 与 AsyncPGDB 实例上验证。
3. 不单独测试私有方法：_start、_missing_tables 等私有成员不做直接断言，
   仅作为 fixture 的初始化入口，其逻辑由公共方法的行为间接验证。
4. 不为父类建测试类：DBEngine、AsyncDBEngine、BaseDB 不设独立测试套件。

PGDB 与 AsyncPGDB 的业务方法需要真实 PostgreSQL，因此整个模块以 integration 标记
隔离：未传 --integration 时整体跳过；异步用例另需安装 asyncpg 驱动。
连接参数取自 conftest 的 db 分区，可用 LOONG_TEST_PG_DSN 覆盖。

运行：
    python -m pytest tests --integration -q
'''

from __future__ import annotations

import importlib.util
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import pytest


try:
    import sqlalchemy
    import pytest_asyncio
    import asyncpg
    from sqlalchemy import MetaData

    from conftest import CONFIG_DB_SETTINGS
    from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine
    from loong_memories.storage.message.pg_db import AsyncPGDB, PGDB
    from loong_memories.storage.message.registry_db import is_db_registered, resolve_db
    from loong_memories.storage.message.schema import (
        HistoryRecord,
        Message,
        get_history_table,
        get_messages_table,
    )

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)


pytestmark = pytest.mark.integration

if '--integration' not in sys.argv:
    pytest.skip(f'[SKIP INFO] File {__file__}：integration 集成测试，默认跳过（请使用 --integration 参数运行）', allow_module_level=True)



# ---------- 常量与辅助 ----------

DSN = 'postgresql://svc:secret@db.example:5433/mem'


def _random_id():
    '''随机正整数：避免跨用例、跨次运行的历史数据互相干扰。'''
    return random.randint(1, 2 ** 31 - 1)



# ---------- fixtures ----------

@pytest.fixture(scope='module')
def db_settings():
    '''同步用例：conftest 的 db 分区深拷贝，后端切到同步 PG。'''
    return CONFIG_DB_SETTINGS.model_copy(deep=True, update={'backend': 'pg', 'table_prefix': 'pg_'})


@pytest_asyncio.fixture(scope='module', loop_scope='module')
def async_db_settings(db_settings):
    '''异步用例：在 db_settings 上切到异步 PG 后端。'''
    return db_settings.model_copy(update={'backend': 'pg_async', 'table_prefix': 'pg_async_'})


class TestPGDB:
    '''PGDB 的公共契约与继承自父类的公共方法。'''

    # ---- backend 类属性 ----

    def test_backend_is_registered(self):
        assert PGDB.backend == 'pg'
        assert is_db_registered('pg') is True
        assert resolve_db('pg') is PGDB

    def test_inherits_business_contract(self):
        assert issubclass(PGDB, DBEngine)
        assert isinstance(PGDB(), DBEngine)

    # ---- __init__ ----

    def test_init_defaults(self):
        db = PGDB()
        assert db.connection_string is None
        assert (db.host, db.port, db.user, db.password, db.database) == ('localhost', 5432, 'postgres', '', '')
        assert (db.print_sql, db.table_prefix) == (False, '')
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (10, 20, 30)
        assert (db.pool_recycle, db.pool_pre_ping) == (1800, True)

    def test_init_stores_conftest_values(self, db_settings):
        pg, pool = db_settings.pg, db_settings.pool_config
        db = PGDB(
            connection_string=DSN,
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.database,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            table_prefix=db_settings.table_prefix,
        )
        assert db.connection_string == DSN
        assert (db.host, db.port, db.user, db.password) == (pg.host, pg.port, pg.user, pg.password)
        assert (db.database, db.print_sql) == (pg.database, pool.print_sql)
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (pool.pool_size, pool.max_overflow, pool.pool_timeout)
        assert (db.pool_recycle, db.pool_pre_ping) == (pool.pool_recycle, pool.pool_pre_ping)
        assert db.table_prefix == db_settings.table_prefix

    def test_init_builds_table_names_from_prefix(self, db_settings):
        db = PGDB(table_prefix=db_settings.table_prefix)
        assert db.history_table.name == db_settings.table_prefix + 'history'
        assert db.messages_table.name == db_settings.table_prefix + 'messages'

    def test_init_leaves_connection_unset(self):
        db = PGDB()
        assert (db.engine, db.SessionLocal) == (None, None)

    def test_enter_returns_self(self):
        db = PGDB()
        assert db.__enter__() is db


class TestAsyncPGDB:
    '''AsyncPGDB 的公共契约与继承自父类的公共方法。'''

    # ---- backend 类属性 ----

    def test_backend_is_registered(self):
        assert AsyncPGDB.backend == 'pg_async'
        assert is_db_registered('pg_async') is True
        assert resolve_db('pg_async') is AsyncPGDB

    def test_inherits_business_contract(self):
        assert issubclass(AsyncPGDB, AsyncDBEngine)
        assert issubclass(AsyncPGDB, DBEngine)

    # ---- __init__ ----

    def test_init_defaults(self):
        db = AsyncPGDB()
        assert db.connection_string is None
        assert (db.host, db.port, db.user, db.password, db.database) == ('localhost', 5432, 'postgres', '', '')
        assert (db.print_sql, db.table_prefix) == (False, '')
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (10, 20, 30)
        assert (db.pool_recycle, db.pool_pre_ping) == (1800, True)

    def test_init_stores_conftest_values(self, async_db_settings):
        pg, pool = async_db_settings.pg, async_db_settings.pool_config
        db = AsyncPGDB(
            connection_string=DSN,
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.database,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
        )
        assert db.connection_string == DSN
        assert (db.host, db.port, db.user, db.password) == (pg.host, pg.port, pg.user, pg.password)
        assert (db.database, db.print_sql) == (pg.database, pool.print_sql)
        assert (db.pool_size, db.max_overflow, db.pool_timeout) == (pool.pool_size, pool.max_overflow, pool.pool_timeout)
        assert (db.pool_recycle, db.pool_pre_ping) == (pool.pool_recycle, pool.pool_pre_ping)

    @pytest.mark.xfail(strict=True, reason='已知缺陷：AsyncDBEngine 构造函数把 table_prefix 重置为空串')
    def test_init_keeps_table_prefix(self):
        assert AsyncPGDB(table_prefix='lm_').table_prefix == 'lm_'

    def test_init_leaves_connection_unset(self):
        db = AsyncPGDB()
        assert (db.engine, db.AsyncSessionLocal) == (None, None)

    def test_enter_returns_self(self):
        db = AsyncPGDB()
        assert db.__enter__() is db


# ---------- 真实数据库：PGDB ----------

class TestPGDBIntegration:
    '''PGDB 在真实 PostgreSQL 上的公共方法行为：历史记录与会话消息的读写淘汰。

    连接参数取自 conftest 的 db 分区，可用 LOONG_TEST_PG_DSN 覆盖；
    表名固定取自 db_settings.table_prefix，模块内共享同一对表；用例靠随机 memory_id 与会话作用域隔离数据。
    '''

    @pytest.fixture(scope='module')
    def pg_db(self, db_settings):
        '''按 conftest 配置建立实例并完成初始化（_start 仅作 fixture 引导）。'''
        dsn = os.environ.get('LOONG_TEST_PG_DSN', '') or db_settings.pg.connection_string
        pg = db_settings.pg
        pool = db_settings.pool_config
        db = PGDB(
            connection_string=dsn or None,
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.database,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            table_prefix=db_settings.table_prefix,
            sslmode=pg.sslmode,
            connect_timeout=pg.connect_timeout,
        )
        try:
            db._start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
        yield db

        db.engine.dispose()

    def test_add_history_assigns_snowflake_id(self, pg_db):
        memory_id = _random_id()
        record = pg_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='hello', event='ADD'))
        assert record.id not in (None, 0)
        assert (record.memory_id, record.new_memory, record.event) == (memory_id, 'hello', 'ADD')
        assert record.is_deleted is False

    def test_add_history_keeps_caller_supplied_id(self, pg_db):
        record_id, memory_id = _random_id(), _random_id()
        record = pg_db.add_history(HistoryRecord(id=record_id, memory_id=memory_id, new_memory='n'))
        assert record.id == record_id
        assert [item.id for item in pg_db.get_history(memory_id)] == [record_id]

    def test_add_history_persists_all_fields(self, pg_db):
        memory_id = _random_id()
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        saved = pg_db.add_history(HistoryRecord(
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
        got = pg_db.get_history(memory_id)[0]
        assert got.id == saved.id
        assert (got.old_memory, got.new_memory, got.event) == ('old', 'new', 'UPDATE')
        assert (got.is_deleted, got.actor_id, got.role) == (True, 'u-1', 'user')

    def test_batch_add_history_empty_list(self, pg_db):
        assert pg_db.batch_add_history([]) == []

    def test_batch_add_history_assigns_unique_ids(self, pg_db):
        memory_id = _random_id()
        records = [HistoryRecord(memory_id=memory_id, new_memory='n' + str(index)) for index in range(3)]
        saved = pg_db.batch_add_history(records)
        assert len(saved) == 3
        assert all(item.id is not None for item in saved)
        assert len({item.id for item in saved}) == 3
        assert len(pg_db.get_history(memory_id)) == 3

    def test_get_history_filters_by_memory_id(self, pg_db):
        first_id, second_id = _random_id(), _random_id()
        pg_db.add_history(HistoryRecord(memory_id=first_id, new_memory='a'))
        pg_db.add_history(HistoryRecord(memory_id=second_id, new_memory='b'))
        assert [item.new_memory for item in pg_db.get_history(first_id)] == ['a']

    def test_get_history_unknown_memory_returns_empty(self, pg_db):
        assert pg_db.get_history(_random_id()) == []

    def test_get_history_returns_desc_by_id(self, pg_db):
        memory_id = _random_id()
        first = pg_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='first'))
        second = pg_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='second'))
        assert [item.id for item in pg_db.get_history(memory_id)] == [second.id, first.id]

    def test_save_messages_empty_list(self, pg_db):
        assert pg_db.save_messages([]) == []

    def test_save_messages_persists_under_limit(self, pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(3)
        ]
        saved = pg_db.save_messages(messages, keep_last=10)
        assert len(saved) == 3
        assert all(item.id is not None for item in saved)
        assert [item.content for item in pg_db.get_last_messages(scope, limit=10)] == ['m0', 'm1', 'm2']

    def test_save_messages_evicts_beyond_keep_last(self, pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(5)
        ]
        pg_db.save_messages(messages, keep_last=2)
        assert [item.content for item in pg_db.get_last_messages(scope, limit=10)] == ['m3', 'm4']

    def test_save_messages_evicts_per_scope(self, pg_db):
        scope_a, scope_b = 'scope-' + str(_random_id()), 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content=str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for scope in (scope_a, scope_b)
            for index in range(3)
        ]
        pg_db.save_messages(messages, keep_last=2)
        assert [item.content for item in pg_db.get_last_messages(scope_a, limit=10)] == ['1', '2']
        assert [item.content for item in pg_db.get_last_messages(scope_b, limit=10)] == ['1', '2']

    def test_get_last_messages_respects_limit(self, pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(5)
        ]
        pg_db.save_messages(messages, keep_last=10)
        assert [item.content for item in pg_db.get_last_messages(scope, limit=2)] == ['m3', 'm4']

    def test_get_last_messages_orders_by_created_at_asc(self, pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(3)
        ]
        pg_db.save_messages(messages, keep_last=10)
        got = pg_db.get_last_messages(scope, limit=10)
        assert all(isinstance(item.created_at_tz, datetime) for item in got)
        assert [item.created_at_tz for item in got] == sorted(item.created_at_tz for item in got)

    def test_get_last_messages_unknown_scope_returns_empty(self, pg_db):
        assert pg_db.get_last_messages('scope-' + str(_random_id()), limit=5) == []


# ---------- 真实数据库：AsyncPGDB ----------

@pytest.mark.asyncio(loop_scope='module')
class TestAsyncPGDBIntegration:
    '''AsyncPGDB 在真实 PostgreSQL 上的公共方法行为，语义与同步版本一致。'''

    @pytest_asyncio.fixture(scope='module', loop_scope='module')
    async def async_pg_db(self, async_db_settings):
        '''建立异步实例并初始化；表名固定取自配置，结束后释放引擎。

        父类构造函数会把 table_prefix 重置为空串（已知缺陷），因此在初始化前
        按传入前缀重建表元数据，避免污染公共 history 与 messages 表。
        '''
        dsn = os.environ.get('LOONG_TEST_PG_DSN', '') or async_db_settings.pg.connection_string
        prefix = async_db_settings.table_prefix
        pg = async_db_settings.pg
        pool = async_db_settings.pool_config
        db = AsyncPGDB(
            connection_string=dsn or None,
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.database,
            print_sql=pool.print_sql,
            pool_size=pool.pool_size,
            max_overflow=pool.max_overflow,
            pool_timeout=pool.pool_timeout,
            pool_recycle=pool.pool_recycle,
            pool_pre_ping=pool.pool_pre_ping,
            table_prefix=prefix,
            sslmode=pg.sslmode,
            connect_timeout=pg.connect_timeout,
        )
        db.table_prefix = prefix
        db.METADATA = MetaData()
        db.history_table = get_history_table(db.METADATA, prefix)
        db.messages_table = get_messages_table(db.METADATA, prefix)
        try:
            await db._start()
        except Exception as exc:
            pytest.skip('无法连接 conftest 配置的 PostgreSQL：' + str(exc))
        yield db

        await db.engine.dispose()

    async def test_add_history_assigns_id_and_persists(self, async_pg_db):
        memory_id = _random_id()
        record = await async_pg_db.add_history(HistoryRecord(memory_id=memory_id, new_memory='hello', event='ADD'))
        assert record.id not in (None, 0)
        assert (record.memory_id, record.event) == (memory_id, 'ADD')
        assert [item.id for item in await async_pg_db.get_history(memory_id)] == [record.id]

    async def test_add_history_keeps_caller_supplied_id(self, async_pg_db):
        record_id, memory_id = _random_id(), _random_id()
        record = await async_pg_db.add_history(HistoryRecord(id=record_id, memory_id=memory_id, new_memory='n'))
        assert record.id == record_id
        assert [item.id for item in await async_pg_db.get_history(memory_id)] == [record_id]

    async def test_batch_add_history_empty_list(self, async_pg_db):
        assert await async_pg_db.batch_add_history([]) == []

    async def test_batch_add_history_assigns_unique_ids(self, async_pg_db):
        memory_id = _random_id()
        records = [HistoryRecord(memory_id=memory_id, new_memory='n' + str(index)) for index in range(3)]
        saved = await async_pg_db.batch_add_history(records)
        assert len(saved) == 3
        assert len({item.id for item in saved}) == 3
        assert len(await async_pg_db.get_history(memory_id)) == 3

    async def test_get_history_filters_by_memory_id(self, async_pg_db):
        first_id, second_id = _random_id(), _random_id()
        await async_pg_db.add_history(HistoryRecord(memory_id=first_id, new_memory='a'))
        await async_pg_db.add_history(HistoryRecord(memory_id=second_id, new_memory='b'))
        assert [item.new_memory for item in await async_pg_db.get_history(first_id)] == ['a']

    async def test_get_history_unknown_memory_returns_empty(self, async_pg_db):
        assert await async_pg_db.get_history(_random_id()) == []

    async def test_save_messages_empty_list(self, async_pg_db):
        assert await async_pg_db.save_messages([]) == []

    async def test_save_messages_persists_under_limit(self, async_pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(3)
        ]
        saved = await async_pg_db.save_messages(messages, keep_last=10)
        assert len(saved) == 3
        assert [item.content for item in await async_pg_db.get_last_messages(scope, limit=10)] == ['m0', 'm1', 'm2']

    async def test_save_messages_evicts_beyond_keep_last(self, async_pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(5)
        ]
        await async_pg_db.save_messages(messages, keep_last=2)
        assert [item.content for item in await async_pg_db.get_last_messages(scope, limit=10)] == ['m3', 'm4']

    async def test_save_messages_evicts_per_scope(self, async_pg_db):
        scope_a, scope_b = 'scope-' + str(_random_id()), 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content=str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for scope in (scope_a, scope_b)
            for index in range(3)
        ]
        await async_pg_db.save_messages(messages, keep_last=2)
        assert [item.content for item in await async_pg_db.get_last_messages(scope_a, limit=10)] == ['1', '2']
        assert [item.content for item in await async_pg_db.get_last_messages(scope_b, limit=10)] == ['1', '2']

    async def test_get_last_messages_respects_limit(self, async_pg_db):
        scope = 'scope-' + str(_random_id())
        created_at = datetime.now(timezone.utc)
        messages = [
            Message(session_scope=scope, role='user', content='m' + str(index),
                    created_at_tz=created_at + timedelta(seconds=index))
            for index in range(5)
        ]
        await async_pg_db.save_messages(messages, keep_last=10)
        assert [item.content for item in await async_pg_db.get_last_messages(scope, limit=2)] == ['m3', 'm4']

    async def test_get_last_messages_unknown_scope_returns_empty(self, async_pg_db):
        assert await async_pg_db.get_last_messages('scope-' + str(_random_id()), limit=5) == []
