from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SQLiteSettings(BaseModel):
    """SQLite 连接参数：path 为库文件路径，必须由使用方显式配置。"""

    model_config = ConfigDict(extra="forbid")

    # 库文件路径；留空表示未配置，实例化时会报错。
    path: str = Field(default="")


class MySQLSettings(BaseModel):
    """MySQL 连接参数。"""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="localhost")
    port: int = Field(default=3306, gt=0)
    user: str = Field(default="root")
    password: str = Field(default="")
    database: str = Field(default="")
    charset: str = Field(default="utf8mb4")


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