# PostgreSQL 实现：同步优先 psycopg3 并回退 psycopg2，异步用 psycopg3 或 asyncpg。
#
# 缺库时仅在启动时报错，模块本身可正常导入。

from __future__ import annotations

import logging
from typing import Any, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from loong_memories.core.exceptions import OptionalDependencyError, StorageError
from loong_memories.storage.message.db_engine import AsyncDBEngine, DBEngine

logger = logging.getLogger(__name__)

try:
    import psycopg

    _PG_DRIVER = "psycopg"
    _PG_ERROR: Optional[Exception] = None
except ImportError as _exc:
    psycopg = None  # type: ignore[assignment]
    try:
        import psycopg2  # type: ignore[no-redef]

        _PG_DRIVER = "psycopg2"
        _PG_ERROR = None
    except ImportError as _exc2:
        psycopg2 = None  # type: ignore[assignment]
        _PG_DRIVER = ""
        _PG_ERROR = _exc2

try:
    import asyncpg

    _ASYNC_PG_DRIVER = "asyncpg"
    _ASYNC_PG_ERROR: Optional[Exception] = None
except ImportError as _exc:
    asyncpg = None  # type: ignore[assignment]
    _ASYNC_PG_DRIVER = ""
    _ASYNC_PG_ERROR = _exc



class PGDB(DBEngine):
    """基于 PostgreSQL 的同步实现：驱动按可用库选择，连接串前缀统一改写。"""

    backend = "pg"

    def __init__(self,
                 connection_string: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "",
                 database: str = "",
                 print_sql: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 pool_pre_ping: bool = True,
                 table_prefix: str = "",
                 sslmode: str = "prefer",
                 connect_timeout: int = 5,
                 application_name: str = "",
                 options: str = "",
                 target_session_attrs: str = "any",
                 ) -> None:

        self.connection_string = connection_string
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.print_sql = print_sql
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.table_prefix = table_prefix

        # 新增的 PG 高级/可选连接参数绑定
        self.sslmode = sslmode
        self.connect_timeout = connect_timeout
        self.application_name = application_name
        self.options = options
        self.target_session_attrs = target_session_attrs
        super().__init__(table_prefix=table_prefix)

    def _start(self) -> Engine:
        """用离散参数或连接串建立引擎，驱动名按可用库选择。"""

        if not _PG_DRIVER:
            raise OptionalDependencyError("使用 PG 数据库需要 psycopg 或 psycopg2") from _PG_ERROR

        parse_connection_string = self.connection_string

        # 连接串里的驱动前缀统一改成当前可用驱动：装了 psycopg3 却写 psycopg2 时也能连。
        if parse_connection_string:
            for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://", "postgres://", "postgresql://"):
                if parse_connection_string.startswith(prefix):
                    parse_connection_string = parse_connection_string.replace(
                        prefix, "postgresql+" + _PG_DRIVER + "://", 1)
                    break
        else:
            url = URL.create(
                "postgresql+" + _PG_DRIVER,
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
                query={
                    "sslmode": self.sslmode,
                    "connect_timeout": str(self.connect_timeout),
                    "application_name": self.application_name,
                    **({"options": self.options} if self.options else {}),
                    "target_session_attrs": self.target_session_attrs,
                }
            )
            parse_connection_string = url.render_as_string(hide_password=False)

        self.engine = create_engine(
            parse_connection_string,
            echo=self.print_sql,
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
        """启动时检查：返回尚未建表的数据表列表（兼容异步引擎，PG 语法）。"""
        all_tables = list(self.METADATA.sorted_tables)
        table_names = [table.name for table in all_tables]

        if not table_names:
            return []

        # 一次查全 information_schema，再把缺失的表挑出来。
        placeholders = ", ".join([":t" + str(index) for index in range(len(table_names))])
        params = {"t" + str(index): name for index, name in enumerate(table_names)}
        query = text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name IN (" + placeholders + ")"
        )

        with self.engine.begin() as conn:
            result = conn.execute(query, params)
            existing = {row[0] for row in result.fetchall()}

        return [table for table in all_tables if table.name not in existing]


class AsyncPGDB(AsyncDBEngine):
    """基于 PostgreSQL 的异步实现：驱动按可用库选择，连接串前缀统一改写。"""

    backend = "pg_async"

    def __init__(self,
                 connection_string: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "",
                 database: str = "",
                 print_sql: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 pool_pre_ping: bool = True,
                 table_prefix: str = "",
                 ) -> None:

        self.connection_string = connection_string
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.print_sql = print_sql
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.table_prefix = table_prefix
        super().__init__(table_prefix=table_prefix)

    async def _start(self) -> AsyncEngine:
        """用离散参数或连接串建立异步引擎，驱动名按可用库选择。"""

        if not _ASYNC_PG_DRIVER:
            raise OptionalDependencyError(
                "使用异步 PG 数据库需要安装 psycopg 或 asyncpg"
            ) from _ASYNC_PG_ERROR

        parse_connection_string = self.connection_string

        # 同步与异步驱动名统一改写成当前异步驱动，避免同一连接串在两种模式下失效。
        if parse_connection_string:
            for prefix in (
                    "postgresql+psycopg2://", "postgresql+psycopg://",
                    "postgresql+asyncpg://", "postgres://", "postgresql://",
            ):
                if parse_connection_string.startswith(prefix):
                    parse_connection_string = parse_connection_string.replace(
                        prefix, "postgresql+" + _ASYNC_PG_DRIVER + "://", 1)
                    break
        else:
            url = URL.create(
                "postgresql+" + _ASYNC_PG_DRIVER,
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
            )
            parse_connection_string = url.render_as_string(hide_password=False)

        try:
            self.engine = create_async_engine(
                parse_connection_string,
                echo=self.print_sql,
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
            raise StorageError("无法创建异步 PG 引擎：" + str(exc)) from exc

        return self.engine

    async def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表（兼容异步引擎，PG 语法）。"""
        all_tables = list(self.METADATA.sorted_tables)
        table_names = [table.name for table in all_tables]

        if not table_names:
            return []

        # 一次查全 information_schema，再把缺失的表挑出来。
        placeholders = ", ".join([":t" + str(index) for index in range(len(table_names))])
        params = {"t" + str(index): name for index, name in enumerate(table_names)}
        query = text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name IN (" + placeholders + ")"
        )

        async with self.engine.begin() as conn:
            result = await conn.execute(query, params)
            existing = {row[0] for row in result.fetchall()}

        return [table for table in all_tables if table.name not in existing]
