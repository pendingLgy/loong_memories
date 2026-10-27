from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SQLiteSettings(BaseModel):
    """SQLite 连接参数：path 为库文件路径，必须由使用方显式配置。"""

    model_config = ConfigDict(extra="forbid")

    # 库文件路径；留空表示未配置，实例化时会报错。
    path: str = Field(default="")

    # ---------------- SQLite 可选连接/驱动参数 ----------------
    timeout: float = Field(
        default=30.0,
        description="数据库写锁等待超时时间（秒），避免并发写入时立即抛出 database is locked"
    )
    isolation_level: str = Field(
        default="",
        description="事务隔离级别：DEFERRED, IMMEDIATE, EXCLUSIVE, AUTOCOMMIT，留空则使用驱动默认"
    )
    check_same_thread: bool = Field(
        default=False,
        description="是否强制检查同线程访问（在异步或多线程连接池场景下通常设为 False）"
    )


class MySQLSettings(BaseModel):
    """MySQL 连接参数。"""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=3306, gt=0)
    user: str = Field(default="root")
    password: str = Field(default="")
    database: str = Field(default="")
    charset: str = Field(default="utf8mb4")

    # ---------------- MySQL 可选连接/驱动参数 ----------------
    connect_timeout: int = Field(
        default=10,
        description="建立数据库连接的超时时间（秒）"
    )
    autocommit: bool = Field(
        default=False,
        description="是否默认开启自动提交事务"
    )
    sql_mode: str = Field(
        default="",
        description="SQL 模式设置，例如: STRICT_TRANS_TABLES,NO_AUTO_CREATE_USER"
    )


class PGSettings(BaseModel):
    """PostgreSQL 连接参数；connection_string 非空时优先于离散字段。"""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=5432, gt=0)
    user: str = Field(default="postgres")
    password: str = Field(default="")
    database: str = Field(default="")
    # 完整连接串；留空则由上面的离散字段拼装。
    connection_string: str = Field(default="")

    # ---------------- 扩展的高级/可选连接参数 ----------------

    sslmode: str = Field(
        default="prefer",
        description="SSL 加密模式：disable, allow, prefer, require, verify-ca, verify-full"
    )
    connect_timeout: int = Field(
        default=5,
        description="建立连接的最大等待时间（秒）"
    )
    application_name: str = Field(
        default="",
        description="客户端应用名称（便于在数据库端监控查询来源）"
    )
    options: str = Field(
        default="",
        description="会话启动参数，例如传递时区或编码：-c timezone=UTC"
    )
    target_session_attrs: str = Field(
        default="any",
        description="高可用/集群路由目标：any（任意）, read-write（仅连接可写主库）"
    )

class DBPoolSettings(BaseModel):
    """数据库连接池"""

    model_config = ConfigDict(extra="forbid")

    print_sql: bool = Field(default=False)
    pool_size: int = Field(default=5, gt=0)
    max_overflow: int = Field(default=10, gt=0)
    pool_timeout: int = Field(default=5, gt=0)
    pool_recycle: int = Field(default=1800, gt=0)
    pool_pre_ping: bool = Field(default=False)


class DBSettings(BaseModel):
    """数据库分区：按 sqlite、mysql、pg 分别承载连接参数。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # 当前使用的后端名，对应 storage.db 注册表中的键；带 _async 后缀者返回异步实现。
    backend: Literal[
        "sqlite", "mysql", "pg",
        "sqlite_async", "mysql_async", "pg_async",
    ] = Field(default="sqlite")

    sqlite: SQLiteSettings = Field(default_factory=SQLiteSettings)
    mysql: MySQLSettings = Field(default_factory=MySQLSettings)
    pg: PGSettings = Field(default_factory=PGSettings)

    table_prefix: str = Field(default="")

    pool_config: DBPoolSettings = Field(default_factory=DBPoolSettings)