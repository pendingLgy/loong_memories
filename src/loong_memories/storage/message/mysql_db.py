# MySQL 实现：优先 PyMySQL 驱动，缺失时回退 mysql-connector-python。
#
# 缺库时仅在实例化时报错，模块本身可正常导入。

from __future__ import annotations

import logging
from typing import Optional, List, Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from loong_memories.core.exceptions import OptionalDependencyError, StorageError
from loong_memories.storage.message.db_engine import DBEngine, AsyncDBEngine

logger = logging.getLogger(__name__)

try:
    import pymysql

    _MYSQL_DRIVER = "pymysql"
    _MYSQL_ERROR: Optional[Exception] = None
except ImportError as _exc:
    pymysql = None  # type: ignore[assignment]
    try:
        import mysql.connector as mysql_connector

        _MYSQL_DRIVER = "mysqlconnector"
        _MYSQL_ERROR = None
    except ImportError as _exc2:
        mysql_connector = None  # type: ignore[assignment]
        _MYSQL_DRIVER = ""
        _MYSQL_ERROR = _exc2


class MySQLDB(DBEngine):
    """基于 MySQL 的实现。"""

    backend = "mysql"

    def __init__(self,
                 connection_string: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 3306,
                 user: str = "root",
                 password: str = "",
                 database: str = "",
                 charset: str = "utf8mb4",
                 print_sql: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 pool_pre_ping: bool = True,
                 table_prefix: str = "",
                 connect_timeout: int = 10,
                 autocommit: bool = False,
                 sql_mode: str = "",
                 ) -> None:

        self.connection_string = connection_string
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.print_sql = print_sql
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.table_prefix = table_prefix

        # 绑定 MySQL 可选驱动参数
        self.connect_timeout = connect_timeout
        self.autocommit = autocommit
        self.sql_mode = sql_mode

        super().__init__(table_prefix=table_prefix)

    def _start(self) -> Engine:
        "用离散参数构造 URL，驱动名按可用库选择。"

        if not _MYSQL_DRIVER:
            raise OptionalDependencyError("使用 MySQL 数据库需要 pymysql 或 mysql-connector-python") from _MYSQL_ERROR

        # 处理 connection_string，支持自动适配异步驱动前缀
        if self.connection_string:
            parse_connection_string = self.connection_string
            for prefix in ("mysql+pymysql://", "mysql+mysqlconnector://", "mysql://"):
                if self.connection_string.startswith(prefix):
                    parse_connection_string = self.connection_string.replace(prefix, f"mysql+{_MYSQL_DRIVER}://",
                                                                             1)
                    break
            self.connection_string = parse_connection_string
        else:
            url = URL.create(
                "mysql+" + _MYSQL_DRIVER,
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
                query={"charset": self.charset}
            )

            self.connection_string = url.render_as_string(hide_password=False)

        connect_args = {
            "connect_timeout": self.connect_timeout,
            "autocommit": self.autocommit,
        }

        # 如果设置了 sql_mode，通过 init_command 在建连时初始化
        if self.sql_mode:
            connect_args["init_command"] = f"SET sql_mode='{self.sql_mode}'"

        try:
            self.engine = create_engine(
                self.connection_string,
                echo=self.print_sql,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
                connect_args=connect_args,
            )

            self.SessionLocal = sessionmaker(bind=self.engine)

            self._init_db(self.engine, self.SessionLocal)
        except Exception as exc:
            raise StorageError(f"无法创建同步 MySQL 引擎：{exc}") from exc

        if self.print_sql:
            # 开启 SQLAlchemy engine 和 pool 的调试日志
            logging.basicConfig()
            logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

        return self.engine

    def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表（同步引擎，MySQL 语法）。"""
        all_tables = list(self.METADATA.sorted_tables)
        table_names = [table.name for table in all_tables]

        if not table_names:
            return []

        with self.engine.begin() as conn:
            # 构造批量查询参数，检查哪些表已经存在于当前数据库中
            placeholders = ", ".join([f":t{i}" for i in range(len(table_names))])
            params = {f"t{i}": name for i, name in enumerate(table_names)}

            query = text(
                f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ({placeholders})"
            )

            result = conn.execute(query, params)
            # 获取当前数据库中已存在的表名集合
            existing_tables = {row[0] for row in result.fetchall()}

        # 过滤出尚未创建的表对象列表
        missing_tables = [table for table in all_tables if table.name not in existing_tables]

        for table in missing_tables:
            logger.info(f"table {table.name} is missing")

        return missing_tables

# 异步 MySQL 驱动尝试导入：优先 aiomysql，缺失时回退 asyncmy
try:
    import aiomysql

    _ASYNC_MYSQL_DRIVER = "aiomysql"
    _ASYNC_MYSQL_ERROR: Optional[Exception] = None
except ImportError as _exc:
    aiomysql = None  # type: ignore[assignment]
    try:
        import asyncmy

        _ASYNC_MYSQL_DRIVER = "asyncmy"
        _ASYNC_MYSQL_ERROR = None
    except ImportError as _exc2:
        asyncmy = None  # type: ignore[assignment]
        _ASYNC_MYSQL_DRIVER = ""
        _ASYNC_MYSQL_ERROR = _exc2


class AsyncMySQLDB(AsyncDBEngine):
    """基于异步 SQLAlchemy 的 MySQL 实现，支持连接池与连接字符串。"""

    backend = "mysql_async"

    def __init__(self,
                 connection_string: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 3306,
                 user: str = "root",
                 password: str = "",
                 database: str = "",
                 charset: str = "utf8mb4",
                 print_sql: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 1800,
                 pool_pre_ping: bool = True,
                 table_prefix: str = "",
                 connect_timeout: int = 10,
                 autocommit: bool = False,
                 sql_mode: str = "",
                 ) -> None:

        self.connection_string = connection_string
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.print_sql = print_sql
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.table_prefix = table_prefix

        # 绑定 MySQL 可选驱动参数
        self.connect_timeout = connect_timeout
        self.autocommit = autocommit
        self.sql_mode = sql_mode

        super().__init__(table_prefix=table_prefix)

    async def _start(self) -> AsyncEngine:
        "用离散参数构造 URL，驱动名按可用库选择。"
        if not _ASYNC_MYSQL_DRIVER:
            raise OptionalDependencyError(
                "使用异步 MySQL 数据库需要安装 aiomysql 或 asyncmy 驱动"
            ) from _ASYNC_MYSQL_ERROR

        parse_connection_string = self.connection_string

        # 处理 connection_string，支持自动适配异步驱动前缀
        if self.connection_string:
            for prefix in ("mysql+pymysql://", "mysql+mysqlconnector://", "mysql://"):
                if self.connection_string.startswith(prefix):
                    parse_connection_string = self.connection_string.replace(prefix, f"mysql+{_ASYNC_MYSQL_DRIVER}://",1)
                    break
        else:
            url = URL.create(
                f"mysql+{_ASYNC_MYSQL_DRIVER}",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
                query={"charset": self.charset}
            )
            # 渲染成字符串链接
            parse_connection_string = url.render_as_string(hide_password=False)

        # 构造传给底层异步 MySQL 驱动的 connect_args
        connect_args = {
            "connect_timeout": self.connect_timeout,
            "autocommit": self.autocommit,
        }

        # 如果设置了 sql_mode，通过 init_command 在异步建连时动态设置
        if self.sql_mode:
            connect_args["init_command"] = f"SET sql_mode='{self.sql_mode}'"

        # 初始化异步引擎与会话工厂
        try:
            self.engine: AsyncEngine = create_async_engine(
                parse_connection_string,
                echo=self.print_sql,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
                connect_args=connect_args,
            )

            self.AsyncSessionLocal = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            await self._init_db(self.engine, self.AsyncSessionLocal)

            if self.print_sql:
                # 开启 SQLAlchemy engine 和 pool 的调试日志
                logging.basicConfig()
                logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

        except Exception as exc:
            raise StorageError(f"无法创建异步 MySQL 引擎：{exc}") from exc

        return self.engine


    async def _missing_tables(self) -> List[Any]:
        """启动时检查：返回尚未建表的数据表列表（兼容异步引擎，MySQL 语法）。"""
        all_tables = list(self.METADATA.sorted_tables)
        table_names = [table.name for table in all_tables]

        if not table_names:
            return []

        async with self.engine.begin() as conn:
            # 构造批量查询参数，检查哪些表已经存在于当前数据库中
            placeholders = ", ".join([f":t{i}" for i in range(len(table_names))])
            params = {f"t{i}": name for i, name in enumerate(table_names)}

            query = text(
                f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN ({placeholders})"
            )

            result = await conn.execute(query, params)
            # 获取当前数据库中已存在的表名集合（统一转小写或保持原样，视 MySQL 配置而定）
            existing_tables = {row[0] for row in result.fetchall()}

        # 过滤出尚未创建的表对象列表
        missing_tables = [table for table in all_tables if table.name not in existing_tables]

        for table in missing_tables:
            logger.info(f"table {table.name} is missing")

        return missing_tables
