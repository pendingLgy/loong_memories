# 关系型数据库通用实现：基于 SQLAlchemy Core，方言差异由引擎适配。

from __future__ import annotations

import logging
from abc import abstractmethod
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Iterator

from sqlalchemy import delete, insert, select, MetaData, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from loong_memories.core.exceptions import ConfigError, StorageError
from loong_memories.storage.message.base import BaseDB
from loong_memories.storage.message.schema import HistoryRecord, Message, get_history_table, get_messages_table
from loong_memories.util.snowflake import next_id
from loong_memories.core.config_db import DBSettings, SQLiteSettings

logger = logging.getLogger(__name__)


class DBEngine(BaseDB):
    """基于 SQLAlchemy Core 的通用实现：单引擎加互斥锁，方言差异由子类提供连接串。"""

    def __init__(self, table_prefix: str = "") -> None:
        self.engine = None
        self.SessionLocal = None
        self.table_prefix = table_prefix

        # 元数据与两张表；建表 DDL 由 SQLAlchemy 按所用方言生成。
        self.METADATA = MetaData()
        self.history_table = get_history_table(self.METADATA, table_prefix)
        self.messages_table = get_messages_table(self.METADATA, table_prefix)

    def _init_db(self, engine, sessionLocal):
        self.engine = engine
        self.SessionLocal = sessionLocal

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                logger.info("SELECT 1")
                assert row[0] == 1
        except Exception as exc:
            logger.error(f"同步数据库连接失败: {exc}")
            raise

        self._ensure_schema()

    @contextmanager
    def _session(self) -> Iterator[Session]:
        """使用标准 sessionmaker 管理会话生命周期"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @abstractmethod
    def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表。"""

    def _ensure_schema(self) -> None:
        """检查表是否存在，缺失的才创建；全部存在时不做任何写操作。"""
        try:
            missing = self._missing_tables()
            if not missing:
                logger.info("tables exist")
                return
            for table in missing:
                logger.info(f"create table {table.name}")
                table.create(self.engine, checkfirst=True)
        except Exception as exc:
            raise StorageError("建表失败" + ": " + str(exc)) from exc

    def _record_row(self, record: HistoryRecord) -> Dict[str, Any]:
        """把历史记录对象转换为可写入的行字典；id 缺省时生成雪花 ID。"""
        return {
            "id": record.id or next_id(),
            "memory_id": record.memory_id,
            "old_memory": record.old_memory,
            "new_memory": record.new_memory,
            "event": record.event,
            "created_at_tz": record.created_at_tz,
            "updated_at_tz": record.updated_at_tz,
            "is_deleted": int(record.is_deleted),
            "actor_id": record.actor_id,
            "role": record.role,
        }

    def _to_record(self, row: Dict[str, Any]) -> HistoryRecord:
        """把数据库行转换为历史记录对象。"""
        return HistoryRecord(
            id=row.get("id"),
            memory_id=row.get("memory_id"),
            old_memory=row.get("old_memory"),
            new_memory=row.get("new_memory"),
            event=row.get("event"),
            created_at_tz=row.get("created_at_tz"),
            updated_at_tz=row.get("updated_at_tz"),
            is_deleted=bool(row.get("is_deleted")),
            actor_id=row.get("actor_id"),
            role=row.get("role"),
        )

    def _message_row(self, message: Message) -> Dict[str, Any]:
        """把消息对象转换为可写入的行字典。"""
        return {
            "id": message.id or next_id(),
            "session_scope": message.session_scope,
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "created_at_tz": message.created_at_tz or datetime.now(timezone.utc).isoformat(),
        }

    def _to_message(self, row: Dict[str, Any]) -> Message:
        """把数据库行转换为消息对象。"""
        return Message(
            id=row.get("id"),
            session_scope=row.get("session_scope"),
            role=row.get("role"),
            content=row.get("content"),
            name=row.get("name"),
            created_at_tz=row.get("created_at_tz"),
        )

    def add_history(self, record: HistoryRecord) -> HistoryRecord:
        """追加一条历史记录，返回补全 id 后的对象。"""
        row = self._record_row(record)
        with self._session() as session:
            session.execute(insert(self.history_table).values(**row))
        return HistoryRecord(**row)

    def batch_add_history(self, records: List[HistoryRecord]) -> List[HistoryRecord]:
        """批量追加历史记录，返回补全 id 后的对象列表。"""
        rows = [self._record_row(record) for record in records]
        if not rows:
            return []
        with self._session() as session:
            session.execute(insert(self.history_table), rows)
        return [HistoryRecord(**row) for row in rows]

    def get_history(self, memory_id: int) -> List[HistoryRecord]:
        """按 memory_id 取历史，按时间升序。"""
        table = self.history_table
        stmt = (
            select(table)
            .where(table.c.memory_id == memory_id)
            .order_by(table.c.id.desc())
        )
        with self._session() as session:
            rows = session.execute(stmt).mappings().all()
        return [self._to_record(dict(row)) for row in rows]

    def save_messages(self, messages: List[Message], keep_last: int = 10) -> List[Message]:
        """写入消息，并只保留各会话最近的 keep_last 条。"""
        rows = [self._message_row(message) for message in messages]
        if not rows:
            return []
        table = self.messages_table
        scopes = sorted({row["session_scope"] for row in rows})
        with self._session() as session:
            session.execute(insert(table), rows)
            for scope in scopes:
                session.execute(self._build_evict_stmt(scope, keep_last))

        return [self._to_message(row) for row in rows]

    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Message]:
        """按时间升序取会话最近 limit 条消息。"""
        table = self.messages_table
        recent = (
            select(
                table.c.id, table.c.session_scope, table.c.role,
                table.c.content, table.c.name, table.c.created_at_tz,
            )
            .where(table.c.session_scope == session_scope)
            .order_by(table.c.id.desc())
            .limit(limit)
            .subquery()
        )
        stmt = select(recent).order_by(recent.c.created_at_tz.asc())
        with self._session() as session:
            rows = session.execute(stmt).mappings().all()
        return [self._to_message(dict(row)) for row in rows]

    def _build_evict_stmt(self, session_scope: str, keep_last: int) -> Any:
        """构造删除语句：保留会话最近 keep_last 条，其余删除。"""
        table = self.messages_table
        recent = (
            select(table.c.id)
            .where(table.c.session_scope == session_scope)
            .order_by(table.c.id.desc())
            .limit(keep_last)
            .subquery()
        )

        stmt = delete(table).where(
            table.c.session_scope == session_scope,
            table.c.id.notin_(select(recent.c.id))
        )

        return stmt

    def __enter__(self) -> "DBEngine":
        return self


class AsyncDBEngine(DBEngine):
    """基于 SQLAlchemy Core 的通用实现：单引擎加互斥锁，方言差异由子类提供连接串。"""

    def __init__(self, table_prefix: str = "") -> None:
        self.engine = None
        self.AsyncSessionLocal = None
        self.table_prefix = table_prefix

        # 元数据与两张表；建表 DDL 由 SQLAlchemy 按所用方言生成。
        self.METADATA = MetaData()
        self.history_table = get_history_table(self.METADATA, table_prefix)
        self.messages_table = get_messages_table(self.METADATA, table_prefix)
        super().__init__()

    async def _init_db(self, engine, asyncSessionLocal):
        self.engine = engine
        self.AsyncSessionLocal = asyncSessionLocal

        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                row = result.fetchone()
                logger.info("SELECT 1")
                assert row[0] == 1
        except Exception as exc:
            logger.error(f"异步数据库连接失败: {exc}")
            raise

        await self._ensure_schema()

    @abstractmethod
    async def _start(self) -> Engine:
        """创建并返回 SQLAlchemy 引擎。"""

    @asynccontextmanager
    async def _session(self) -> Iterator[AsyncSession]:
        """使用标准 sessionmaker 管理会话生命周期"""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @abstractmethod
    async def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表（兼容异步引擎）。"""

    async def _ensure_schema(self) -> None:
        """检查表是否存在，缺失的才创建；全部存在时不做任何写操作。"""
        try:
            missing = await self._missing_tables()
            if not missing:
                logger.info("tables exist")
                return
            async with self.engine.begin() as conn:
                for table in missing:
                    logger.info(f"create table {table.name}")
                    await conn.run_sync(table.create, checkfirst=True)

        except Exception as exc:
            raise StorageError("建表失败" + ": " + str(exc)) from exc

    # ---------------- 历史记录 ----------------

    async def add_history(self, record: HistoryRecord) -> HistoryRecord:
        """追加一条历史记录，返回补全 id 后的对象。"""
        row = self._record_row(record)
        async with self._session() as session:
            await session.execute(insert(self.history_table).values(**row))
        return HistoryRecord(**row)

    async def batch_add_history(self, records: List[HistoryRecord]) -> List[HistoryRecord]:
        """批量追加历史记录，返回补全 id 后的对象列表。"""
        rows = [self._record_row(record) for record in records]
        if not rows:
            return []
        async with self._session() as session:
            await session.execute(insert(self.history_table), rows)
        return [HistoryRecord(**row) for row in rows]

    async def get_history(self, memory_id: int) -> List[HistoryRecord]:
        """按 memory_id 取历史，按时间升序。"""
        table = self.history_table
        stmt = (
            select(table)
            .where(table.c.memory_id == memory_id)
            .order_by(table.c.id.desc())
        )
        async with self._session() as session:
            result = await session.execute(stmt)
            rows = result.mappings().all()
        return [self._to_record(dict(row)) for row in rows]

    # ---------------- 会话消息 ----------------

    async def save_messages(self, messages: List[Message], keep_last: int = 10) -> List[Message]:
        """写入消息，并只保留各会话最近的 keep_last 条。"""
        rows = [self._message_row(message) for message in messages]
        if not rows:
            return []
        table = self.messages_table
        scopes = sorted({row["session_scope"] for row in rows})
        async with self._session() as session:
            await session.execute(insert(table), rows)
            for scope in scopes:
                await session.execute(self._build_evict_stmt(scope, keep_last))

        return [self._to_message(row) for row in rows]

    async def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Message]:
        """按时间升序取会话最近 limit 条消息。"""
        table = self.messages_table
        recent = (
            select(
                table.c.id, table.c.session_scope, table.c.role,
                table.c.content, table.c.name, table.c.created_at_tz,
            )
            .where(table.c.session_scope == session_scope)
            .order_by(table.c.created_at_tz.desc(), table.c.id.desc())
            .limit(limit)
            .subquery()
        )
        stmt = select(recent).order_by(recent.c.created_at_tz.asc())
        async with self._session() as session:
            result = await session.execute(stmt)
            rows = result.mappings().all()
        return [self._to_message(dict(row)) for row in rows]

    def __enter__(self) -> "AsyncDBEngine":
        return self


# 按配置装配数据库实例：按 backend 选择方言，参数取自对应方言分区。
# 方言实现按需惰性导入，避免与本模块形成导入环。
def build_database(settings: "DBSettings") -> BaseDB:
    """ 按 DBSettings.backend 构造对应方言的实例；异步后端的实例需调用方 await _start() 启动。"""

    backend = settings.backend
    pool = settings.pool_config

    # 1. 提取所有后端通用的连接池与表前缀配置，消除冗余
    common_kwargs = {
        "print_sql": pool.print_sql,
        "pool_size": pool.pool_size,
        "max_overflow": pool.max_overflow,
        "pool_timeout": pool.pool_timeout,
        "pool_recycle": pool.pool_recycle,
        "pool_pre_ping": pool.pool_pre_ping,
        "table_prefix": settings.table_prefix,
    }

    # 2. SQLite 后端（合并同步与异步）
    if backend in ("sqlite", "sqlite_async"):
        from loong_memories.storage.message.sqlite_db import SQLiteDB, AsyncSQLiteDB
        db_cls = AsyncSQLiteDB if backend.endswith("_async") else SQLiteDB
        return db_cls(db_path=_sqlite_path(settings.sqlite), **common_kwargs)

    # 3. MySQL 后端（合并同步与异步）
    if backend in ("mysql", "mysql_async"):
        from loong_memories.storage.message.mysql_db import MySQLDB, AsyncMySQLDB
        mysql = settings.mysql
        db_cls = AsyncMySQLDB if backend.endswith("_async") else MySQLDB
        return db_cls(
            host=mysql.host,
            port=mysql.port,
            user=mysql.user,
            password=mysql.password,
            database=mysql.database,
            charset=mysql.charset,
            **common_kwargs,
        )

    # 4. PostgreSQL 后端（合并同步与异步）
    if backend in ("pg", "pg_async"):
        from loong_memories.storage.message.pg_db import PGDB, AsyncPGDB
        pg = settings.pg
        db_cls = AsyncPGDB if backend.endswith("_async") else PGDB
        return db_cls(
            connection_string=pg.connection_string or None,
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.database,
            **common_kwargs,
        )

    raise ConfigError("不支持的数据库后端：" + str(backend))


def _sqlite_path(settings: "SQLiteSettings") -> str:
    """取 SQLite 库文件路径；未配置时统一报错。"""
    if not settings.path:
        raise ConfigError("未配置 SQLite 库文件路径：请设置 db.sqlite.path 或环境变量 RAG_DB__SQLITE__PATH")
    return settings.path
