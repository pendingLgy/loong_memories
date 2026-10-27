from __future__ import annotations

import json
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Iterator, List, Optional, AsyncIterator


from loong_memories.core.exceptions import OptionalDependencyError, StorageError
from loong_memories.storage.memories.base import MetricType, VectorStoreBase, VectorStoreResult

logger = logging.getLogger(__name__)

try:

    from sqlalchemy import create_engine, cast, Numeric, desc, URL, inspect
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy import Table, Column, BigInteger, Text, MetaData, select, insert, update, delete, func, and_, or_, \
    not_, \
    text, Engine, Select, Delete, Update
    from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
    from sqlalchemy.exc import SQLAlchemyError
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine

    _SQLALCHEMY_AVAILABLE = True
    _SQLALCHEMY_ERROR = None
except ImportError as _exc:
    logger.error(str(_exc))
    _SQLALCHEMY_AVAILABLE = False
    _SQLALCHEMY_ERROR = _exc

# -------------------- 具名常量定义 --------------------

DISKANN_MAX_DIMENSION = 2000

# 全文检索表达式
_TEXT_VECTOR = "to_tsvector('simple', text_lemma)"

DIRECT_COLUMNS = {
    "id", "tenant_id", "user_id", "vector",
    "text", "text_lemma",
    "updated_at_tz", "created_at_tz",
}


def _build_sqlalchemy_filter(table: Table, filters: Optional[Dict[str, Any]]) -> Any:
    """
    使用 SQLAlchemy 表达式构建 WHERE 条件，自动适配顶层列与 metadata JSONB 字段。
    """
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        is_direct = key in DIRECT_COLUMNS

        # 对于 metadata 字段，默认转为 string 表达（相当于 metadata->>key）
        col = table.c[key] if is_direct else table.c.metadata[key].as_string()

        # 1. 检查键是否存在 (*)
        if value == "*":
            if is_direct:
                conditions.append(table.c[key].isnot(None))
            else:
                conditions.append(table.c.metadata.has_key(key))
            continue

        # 2. 字典形式的操作符
        if isinstance(value, dict):
            for op, op_val in value.items():
                # 判断是否需要按数值处理（大小比较 且 传入的值是数字类型）
                is_numeric_op = op in ("gt", "gte", "lt", "lte")
                if is_numeric_op and isinstance(op_val, (int, float)) and not is_direct:
                    # 使用 sa.cast 替代 as_numeric()，避免缺少 precision/scale 参数报错
                    expr = cast(table.c.metadata[key], Numeric)
                else:
                    expr = col

                if op == "eq":
                    conditions.append(expr == op_val)
                elif op == "ne":
                    conditions.append(expr != op_val)
                elif op == "gt":
                    conditions.append(expr > op_val)
                elif op == "gte":
                    conditions.append(expr >= op_val)
                elif op == "lt":
                    conditions.append(expr < op_val)
                elif op == "lte":
                    conditions.append(expr <= op_val)
                elif op == "in":
                    if not isinstance(op_val, list):
                        raise ValueError('操作符 in 需要列表值')
                    conditions.append(col.in_(op_val))
                elif op == "nin":
                    if not isinstance(op_val, list):
                        raise ValueError('操作符 nin 需要列表值')
                    conditions.append(~col.in_(op_val))
                elif op == "contains":
                    conditions.append(col.like(f"%{op_val}%"))
                elif op == "icontains":
                    conditions.append(col.ilike(f"%{op_val}%"))
                else:
                    raise ValueError(f'不支持的过滤操作符: {op}')

        # 3. 列表形式 (IN 查询)
        elif isinstance(value, list):
            conditions.append(col.in_(value))

        # 4. 普通等于匹配
        else:
            val_to_compare = json.dumps(value) if isinstance(value, bool) else value
            conditions.append(col == val_to_compare)

    return and_(*conditions) if conditions else None


class PGVectorStore(VectorStoreBase):
    "基于 PostgreSQL、pgvector 与 SQLAlchemy 的向量库实现；表即集合。"

    backend = "pg"
    metric_type = "COSINE"

    def __init__(
            self,
            connection_string: Optional[str] = None,
            collection_name: str = "",
            vector_size: int = 1024,
            database: str = "postgres",
            user: str = "postgres",
            password: str = "",
            host: str = "localhost",
            port: int = 5432,
            use_hnsw: bool = True,
            use_diskann: bool = False,
            print_sql: bool = False,
            pool_size: int = 10,
            max_overflow: int = 20,
            pool_timeout: int = 30,
            pool_recycle: int = 1800,
            pool_pre_ping: bool = True,
    ) -> None:

        self.connection_string = connection_string
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.use_hnsw = use_hnsw
        self.use_diskann = use_diskann
        self.print_sql = print_sql
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping

        self.metadata_obj = MetaData()


    def start(self) -> Any:
        if not _SQLALCHEMY_AVAILABLE:
            raise OptionalDependencyError(
                '使用 SQLAlchemy PG 向量库需要安装 sqlalchemy 及 pgvector，请执行 pip install sqlalchemy pgvector'
            ) from _SQLALCHEMY_ERROR

        if self.connection_string:
            self.connection_string = self.connection_string
        else:
            url = URL.create(
                "postgresql",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
            )
            self.connection_string = url.render_as_string(hide_password=False)

        # 初始化 SQLAlchemy 引擎与元数据
        try:
            self.engine: Engine = create_engine(
                self.connection_string,
                echo=self.print_sql,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
            )

            if self.print_sql:
                # 开启 SQLAlchemy engine 和 pool 的调试日志
                logging.basicConfig()
                logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

            self.SessionLocal = sessionmaker(bind=self.engine)
        except Exception as exc:
            raise StorageError('无法创建 SQLAlchemy 引擎：' + str(exc)) from exc

        self.create_col(self.collection_name, self.vector_size)
        return self

    # ---------------- 辅助方法 ----------------

    def _build_sql_keyword_search(self, filters: dict[str, Any] | None, query: str, top_k: int) -> Select:

        table = self._get_table()
        query_expression = query.replace(" ", " | ")
        ts_query = func.to_tsquery('simple', query_expression)
        ts_vector = func.to_tsvector('simple', table.c.text_lemma)
        score_expr = func.ts_rank_cd(ts_vector, ts_query).label("score")

        stmt = select(
            table.c.id, table.c.tenant_id, table.c.user_id,
            table.c.text, table.c.text_lemma,
            table.c.updated_at_tz, table.c.created_at_tz,
            score_expr, table.c.metadata
        ).where(ts_vector.bool_op('@@')(ts_query))

        where_clause = _build_sqlalchemy_filter(table, filters)
        if where_clause is not None:
            stmt = stmt.where(where_clause)
        stmt = stmt.order_by(desc("score")).limit(top_k)
        return stmt

    def _build_sql_search(self, filters: dict[str, Any] | None, top_k: int, vectors: list[float | int]) -> Select:
        table = self._get_table()
        score_expr = table.c.vector.cosine_distance(vectors).label("score")
        stmt = select(
            table.c.id,
            table.c.tenant_id,
            table.c.user_id,
            table.c.text,
            table.c.text_lemma,
            table.c.updated_at_tz,
            table.c.created_at_tz,
            score_expr,
            table.c.metadata)

        where_clause = _build_sqlalchemy_filter(table, filters)
        if where_clause is not None:
            stmt = stmt.where(where_clause)

        stmt = stmt.order_by("score").limit(top_k)
        return stmt


    def _build_sql_insert(self, ids: list[int], payloads: list[dict[str, Any]], vectors: list[list[float | int]]) -> list:
        rows = [
            {
                "id": rid,
                "tenant_id": item.get("tenant_id"),
                "user_id": item.get("user_id"),
                "vector": vector,
                "metadata": item.get("metadata"),
                "text": item.get("text"),
                "text_lemma": item.get("text_lemma"),
                "updated_at_tz": item.get("updated_at_tz"),
                "created_at_tz": item.get("created_at_tz"),
            }
            for rid, vector, item in zip(ids, vectors, payloads)
        ]
        return rows

    def _build_sql_delete(self, id: int) -> Delete:
        table = self._get_table()
        stmt = delete(table).where(table.c.id == id)
        return stmt

    def _build_sql_update(self, id: int, payload: dict[str, Any] | None,
                          vector: list[float | int] | None) -> Update:

        table = self._get_table()

        update_data = {}
        if vector is not None:
            update_data["vector"] = vector
        if payload is not None:
            update_data["metadata"] = payload

        stmt = update(table).where(table.c.id == id).values(**update_data)
        return stmt

    def _build_sql_get(self, id: int) -> Select[Any, Any, Any, Any, Any, Any, Any, Any] | Select[*tuple[Any, ...]]:
        table = self._get_table()
        stmt = (select(
            table.c.id,
            table.c.tenant_id,
            table.c.user_id,
            table.c.text,
            table.c.text_lemma,
            table.c.updated_at_tz,
            table.c.created_at_tz,
            table.c.metadata
        ).where(table.c.id == id))
        return stmt


    def _build_sql_list(self, filters: dict[str, Any] | None, top_k: int | None) -> Select:
        table = self._get_table()
        stmt = select(table.c.id,
                      table.c.tenant_id,
                      table.c.user_id,
                      table.c.text,
                      table.c.text_lemma,
                      table.c.updated_at_tz,
                      table.c.created_at_tz,
                      table.c.metadata)

        where_clause = _build_sqlalchemy_filter(table, filters)
        if where_clause is not None:
            stmt = stmt.where(where_clause)

        stmt = stmt.limit(top_k or 100)
        return stmt

    def _col(self) -> str:
        "返回经校验的安全表名。"
        if not self.collection_name.replace('_', '').isalnum():
            raise StorageError('非法集合名：' + str(self.collection_name))
        return self.collection_name

    def _get_table(self) -> Table:
        """动态获取当前集合对应的 SQLAlchemy Table 对象。"""
        table_name = self._col()
        if table_name in self.metadata_obj.tables:
            return self.metadata_obj.tables[table_name]

        return Table(
            table_name,
            self.metadata_obj,
            Column("id", BigInteger, primary_key=True),
            Column("tenant_id", BigInteger, nullable=False),
            Column("user_id", BigInteger, nullable=False),
            Column("vector", Vector(self.vector_size)),
            Column("metadata", JSONB),
            Column("text", Text),
            Column("text_lemma", Text),
            Column("updated_at_tz", TIMESTAMP(timezone=True)),
            Column("created_at_tz", TIMESTAMP(timezone=True)),
            extend_existing=True
        )

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

    # ---------------- 集合管理 ----------------

    def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        """同步方法：建表并初始化向量索引与全文索引；若表已存在则直接跳过后续所有操作。"""
        table = self._get_table()

        with self.engine.begin() as conn:

            # 检查表是否存在（如果存在返回表名，不存在返回 None）
            res = conn.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": table.name}
            ).scalar()

            if res is not None:
                logger.info(f"table {table.name} is exist")
                return

            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            table.create(conn, checkfirst=True)

            safe_name = self._col()
            if self.use_diskann and vector_size < DISKANN_MAX_DIMENSION:
                res = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'")).fetchone()
                if res:
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {safe_name}_diskann_idx ON {safe_name} USING diskann (vector)"))
            elif self.use_hnsw:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {safe_name}_hnsw_idx ON {safe_name} USING hnsw (vector vector_cosine_ops)"))

            conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {safe_name}_text_idx ON {safe_name} USING gin ({_TEXT_VECTOR})"))

    # ---------------- 写入与检索 ----------------

    def insert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[int]) -> None:
        "批量写入。"
        table = self._get_table()
        rows = self._build_sql_insert(ids, payloads, vectors)
        with self._session() as session:
            session.execute(insert(table), rows)

    def _parse(self, row_dict: Dict[str, Any]) -> VectorStoreResult:
        # 将时间字段转换为字符串（若存在且不为 None）
        if row_dict.get("created_at_tz") is not None:
            row_dict["created_at_tz"] = row_dict["created_at_tz"].isoformat()

        if row_dict.get("updated_at_tz") is not None:
            row_dict["updated_at_tz"] = row_dict["updated_at_tz"].isoformat()
        store = VectorStoreResult.to_vector_store_result(**row_dict)
        return store

    def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        stmt = self._build_sql_search(filters, top_k, vectors)

        with self._session() as session:
            rows = session.execute(stmt).fetchall()

        results: List[VectorStoreResult] = []
        for row in rows:
            row_dict = row._asdict()
            results.append(self._parse(row_dict))
        return results

    def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        "基于 text_lemma 的全文检索；失败返回 None。"

        stmt = self._build_sql_keyword_search(filters, query, top_k)

        try:
            with self._session() as session:
                rows = session.execute(stmt).fetchall()
        except Exception as _exc:
            import traceback
            # 获取完整的堆栈字符串
            error_msg = traceback.format_exc()
            logger.error(f"操作执行失败，详细堆栈信息如下：\n{error_msg}")
            return None

        results: List[VectorStoreResult] = []
        for row in rows:
            row_dict = row._asdict()
            results.append(self._parse(row_dict))
        return results

    def delete(self, id: int) -> None:
        stmt = self._build_sql_delete(id)
        with self._session() as session:
            session.execute(stmt)


    def update(self, id: int, vector: Optional[List[float]] = None,payload: Optional[Dict[str, Any]] = None) -> None:
        if vector is None and payload is None:
            logger.warning("param is null")
            return

        stmt = self._build_sql_update(id, payload, vector)

        with self._session() as session:
            session.execute(stmt)


    def get(self, id: int) -> Optional[VectorStoreResult]:
        stmt = self._build_sql_get(id)
        with self._session() as session:
            row = session.execute(stmt).fetchone()
        if not row:
            return None
        return self._parse(row._asdict())

    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> Optional[List[VectorStoreResult]]:
        stmt = self._build_sql_list(filters, top_k)
        with self._session() as session:
            rows = session.execute(stmt).fetchall()

        results: List[VectorStoreResult] = []
        for row in rows:
            row_dict = row._asdict()
            results.append(self._parse(row_dict))
        return results




class AsyncPGVectorStore(PGVectorStore):
    """基于 PostgreSQL、pgvector 与 SQLAlchemy Asyncio 的异步向量库实现；表即集合。"""

    backend = "pg_async"
    metric_type = "COSINE"

    def __init__(
            self,
            connection_string: Optional[str] = None,
            collection_name: str = "",
            vector_size: int = 1024,
            database: str = "postgres",
            user: str = "postgres",
            password: str = "",
            host: str = "localhost",
            port: int = 5432,
            use_hnsw: bool = True,
            use_diskann: bool = False,
            print_sql: bool = False,
            pool_size: int = 10,
            max_overflow: int = 20,
            pool_timeout: int = 30,
            pool_recycle: int = 1800,
            pool_pre_ping: bool = True,
    ) -> None:

        self.connection_string = connection_string
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.database = database
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.use_hnsw = use_hnsw
        self.use_diskann = use_diskann
        self.print_sql = print_sql
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping

        self.metadata_obj = MetaData()


    async def start(self) -> Any:
        if not _SQLALCHEMY_AVAILABLE:
            raise OptionalDependencyError(
                '使用异步 SQLAlchemy PG 向量库需要安装 sqlalchemy、asyncpg 及 pgvector'
            ) from _SQLALCHEMY_ERROR

        if self.connection_string:
            parse_connection_string = self.connection_string
            if self.connection_string.startswith("postgresql://"):
                parse_connection_string = self.connection_string.replace("postgresql://", "postgresql+asyncpg://", 1)
            self.connection_string = parse_connection_string
        else:
            url = URL.create(
                "postgresql+asyncpg",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
            )
            self.connection_string = url.render_as_string(hide_password=False)

        try:
            self.engine: AsyncEngine = create_async_engine(
                self.connection_string,
                echo=self.print_sql,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=self.pool_pre_ping,
            )

            if self.print_sql:
                # 开启 SQLAlchemy engine 和 pool 的调试日志
                logging.basicConfig()
                logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)

            self.AsyncSessionLocal = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

        except Exception as exc:
            raise StorageError('无法创建异步 SQLAlchemy 引擎：' + str(exc)) from exc

        await self.create_col(self.collection_name, self.vector_size)

        return self


    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        table = self._get_table()
        async with self.engine.begin() as conn:

            # 检查表是否存在（如果存在返回表名，不存在返回 None）
            res = (await conn.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": table.name}
            )).scalar()

            if res is not None:
                logger.info(f"table {table.name} is exist")
                return

            # 2. 表不存在，开始创建扩展与表
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            await conn.run_sync(table.create, checkfirst=True)

            safe_name = self._col()
            if self.use_diskann and vector_size < DISKANN_MAX_DIMENSION:
                res = await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'"))
                if res.fetchone():
                    await conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {safe_name}_diskann_idx ON {safe_name} USING diskann (vector)"))
            elif self.use_hnsw:
                await conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {safe_name}_hnsw_idx ON {safe_name} USING hnsw (vector vector_cosine_ops)"))

            await conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {safe_name}_text_idx ON {safe_name} USING gin ({_TEXT_VECTOR})"))

    async def insert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[int]) -> None:
        table = self._get_table()
        rows = self._build_sql_insert(ids, payloads, vectors)

        async with self._session() as session:
            await session.execute(insert(table), rows)

    async def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:

        stmt = self._build_sql_search(filters, top_k, vectors)

        async with self._session() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        return [self._parse(row._asdict()) for row in rows]

    async def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        stmt = self._build_sql_keyword_search(filters, query, top_k)

        try:
            async with self._session() as session:
                result = await session.execute(stmt)
                rows = result.fetchall()
        except Exception:
            import traceback
            logger.error(f"异步全文检索执行失败：\n{traceback.format_exc()}")
            return None

        return [self._parse(row._asdict()) for row in rows]


    async def delete(self, id: int) -> None:
        stmt = self._build_sql_delete(id)
        async with self._session() as session:
            await session.execute(stmt)

    async def update(self, id: int, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:

        if vector is None and payload is None:
            logger.warning("param is null")
            return

        stmt = self._build_sql_update(id, payload, vector)

        async with self._session() as session:
            await session.execute(stmt)

    async def get(self, id: int) -> Optional[VectorStoreResult]:
        stmt = self._build_sql_get(id)
        async with self._session() as session:
            result = await session.execute(stmt)
            row = result.fetchone()
        if not row:
            return None
        return self._parse(row._asdict())


    async def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> Optional[List[VectorStoreResult]]:
        stmt = self._build_sql_list(filters, top_k)
        async with self._session() as session:
            result = await session.execute(stmt)
            rows = result.fetchall()

        results: List[VectorStoreResult] = []
        for row in rows:
            row_dict = row._asdict()
            results.append(self._parse(row_dict))
        return results
