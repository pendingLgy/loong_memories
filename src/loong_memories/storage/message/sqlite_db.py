# SQLite 实现：同步走标准库 sqlite3，异步走 aiosqlite（需额外安装）。
#
# 数据一律落盘：库文件路径必须由构造参数显式指定，不提供内存库模式。

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from loong_memories.core.exceptions import ConfigError, OptionalDependencyError, StorageError
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine

logger = logging.getLogger(__name__)

try:
    import aiosqlite

    _ASYNC_SQLITE_DRIVER = "aiosqlite"
    _ASYNC_SQLITE_ERROR: Optional[Exception] = None
except ImportError as _exc:
    aiosqlite = None  # type: ignore[assignment]
    _ASYNC_SQLITE_DRIVER = ""
    _ASYNC_SQLITE_ERROR = _exc


def _ensure_parent(path: str) -> None:
    """确保库文件所在目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


class SQLiteDB(DBEngine):
    """基于 SQLite 的同步实现；db_path 为必填参数，指向库文件路径。"""

    backend = "sqlite"

    def __init__(self,
                 db_path: str,
                 print_sql: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 pool_pre_ping: bool = True,
                 table_prefix: str = "",
                 ) -> None:
        self.db_path = db_path
        self.print_sql = print_sql
        self.connection_string: Optional[str] = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.table_prefix = table_prefix
        super().__init__(table_prefix=table_prefix)

    def _start(self) -> Engine:
        """补齐库文件目录后建立引擎与会话工厂，随后确保表存在。"""

        if not self.db_path:
            raise ConfigError("未配置 SQLite 库文件路径：请设置 db.sqlite.path 或环境变量 RAG_DB__SQLITE__PATH")

        _ensure_parent(self.db_path)

        url = URL.create("sqlite", database=self.db_path)
        self.connection_string = url.render_as_string(hide_password=False)

        # 关闭线程检查：解析与记忆流程可能在不同线程复用同一个库文件。
        self.engine = create_engine(
            url,
            echo=self.print_sql,
            connect_args={
                "check_same_thread": False,
                "timeout": 5.0  # 增加锁等待超时时间（秒）
            },
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=self.pool_pre_ping,
        )

        self.SessionLocal = sessionmaker(bind=self.engine)

        self._init_db(self.engine, self.SessionLocal)

        if self.print_sql:
            # 开启 SQLAlchemy engine 和 pool 的调试日志
            logging.basicConfig()
            logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

        return self.engine

    def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表（SQLite 用通用 inspect 即可）。"""

        all_tables = list(self.METADATA.sorted_tables)
        table_names = [table.name for table in all_tables]

        placeholders = ", ".join([f":t{i}" for i in range(len(table_names))])
        params = {f"t{i}": name for i, name in enumerate(table_names)}

        query_sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({placeholders})"

        with self.engine.connect() as conn:
            result = conn.execute(text(query_sql), params)
            existing_tables = {row[0] for row in result.fetchall()}

        return [table for table in all_tables if table.name not in existing_tables]


class AsyncSQLiteDB(AsyncDBEngine):
    """基于 SQLite 的异步实现；db_path 为必填参数，需安装 aiosqlite 驱动。"""

    backend = "sqlite_async"

    def __init__(self,
                 db_path: str,
                 print_sql: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 pool_pre_ping: bool = True,
                 table_prefix: str = "",
                 ) -> None:
        self.db_path = db_path
        self.print_sql = print_sql
        self.connection_string: Optional[str] = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.table_prefix = table_prefix
        super().__init__(table_prefix=table_prefix)

    async def _start(self) -> AsyncEngine:
        """补齐库文件目录后建立异步引擎与会话工厂，随后确保表存在。"""

        if not _ASYNC_SQLITE_DRIVER:
            raise OptionalDependencyError("使用异步 SQLite 数据库需要安装 aiosqlite") from _ASYNC_SQLITE_ERROR

        if not self.db_path:
            raise ConfigError("未配置 SQLite 库文件路径：请设置 db.sqlite.path 或环境变量 RAG_DB__SQLITE__PATH")

        _ensure_parent(self.db_path)

        url = URL.create("sqlite+" + _ASYNC_SQLITE_DRIVER, database=self.db_path)
        self.connection_string = url.render_as_string(hide_password=False)

        try:
            self.engine = create_async_engine(
                url,
                echo=self.print_sql,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 5.0  # 增加锁等待超时时间（秒）
                },
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
            )

            self.session_local = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            await self._init_db(self.engine, self.session_local)

            if self.print_sql:
                # 开启 SQLAlchemy engine 和 pool 的调试日志
                logging.basicConfig()
                logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

        except Exception as exc:
            raise StorageError("无法创建异步 SQLite 引擎：" + str(exc)) from exc

        return self.engine

    async def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表（SQLite 用通用 inspect 即可）。"""

        all_tables = list(self.METADATA.sorted_tables)
        table_names = [table.name for table in all_tables]

        placeholders = ", ".join([f":t{i}" for i in range(len(table_names))])
        params = {f"t{i}": name for i, name in enumerate(table_names)}

        query_sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({placeholders})"

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query_sql), params)
            existing_tables = {row[0] for row in result.fetchall()}

        return [table for table in all_tables if table.name not in existing_tables]
