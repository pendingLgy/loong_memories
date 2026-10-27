from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from loong_memories.core.config_db import DBPoolSettings


class MilvusSettings(BaseModel):
    """Milvus 连接参数；留空的字段交由实现层使用自身默认值。"""

    model_config = ConfigDict(extra="forbid")

    # 服务地址；留空时用实现默认（本机 19530 端口）。
    uri: str = Field(default="")
    # 鉴权令牌；留空表示不鉴权。
    token: str = Field(default="")
    # 数据库名；留空表示使用 Milvus 默认库。
    db_name: str = Field(default="")

class PGVectorSettings(BaseModel):
    """PostgreSQL 向量库参数；connection_string 非空时优先于离散字段。"""

    model_config = ConfigDict(extra="forbid")

    connection_string: str = Field(default="")
    host: str = Field(default="localhost")
    port: int = Field(default=5432, gt=0)
    user: str = Field(default="postgres")
    password: str = Field(default="")
    database: str = Field(default="")
    # 是否为向量列建立 HNSW 索引。
    use_hnsw: bool = Field(default=True)
    # 是否尝试 DiskANN 索引（需 vectorscale 扩展；缺失时自动回退 HNSW）。
    use_diskann: bool = Field(default=False)

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

class VectorSettings(BaseModel):
    """向量库分区：按 milvus、pg 承载连接参数，并统一集合与检索默认值。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # 当前使用的向量后端名，对应 storage.vector_store 注册表中的键。
    backend: Literal["milvus", "pg"] = Field(default="milvus")
    # 集合名与向量维度。
    collection: str = Field(default="")
    vector_size: int = Field(default=1024, gt=0)
    # 检索默认返回条数。
    top_k: int = Field(default=5, gt=0)
    # 距离度量类型。
    metric_type: Literal["COSINE", "L2", "IP"] = Field(default="COSINE")

    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    pg: PGVectorSettings = Field(default_factory=PGVectorSettings)

    pool_config: DBPoolSettings = Field(default_factory=DBPoolSettings)