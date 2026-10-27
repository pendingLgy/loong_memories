# Milvus 向量库实现：集合、向量字段与 JSON 元数据，AUTOINDEX 建索引。
#
# 单一 schema：id 主键、vectors 稠密向量、metadata 元数据，外加 text 文本与 sparse 稀疏向量；
# 稀疏向量由 BM25 函数按 text 自动生成，稠密检索与关键词检索共用同一集合。

from __future__ import annotations

import json
import logging
import re
from copy import copy
from typing import Any, Dict, List, Optional

from loong_memories.core.exceptions import OptionalDependencyError, StorageError
from loong_memories.storage.memories.base import MetricType, VectorStoreBase, VectorStoreResult

try:  # 可选依赖：模块仍可导入，缺库时仅在实例化时报错。
    from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient
    _PYMILVUS_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover
    CollectionSchema = DataType = FieldSchema = Function = FunctionType = MilvusClient = None  # type: ignore[assignment]
    _PYMILVUS_ERROR = exc

# Milvus 过滤表达式的键名白名单，避免表达式注入。
_SAFE_FILTER_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
# 文本字段最大长度（Milvus VARCHAR 上限）。
_MAX_TEXT_LEN = 65535


logger = logging.getLogger(__name__)


def _as_scope_int(value: Any) -> Optional[int]:
    """作用域标识归一为 INT64：None 与空串视为未设置，无法转整数时告警并丢弃。"""
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning('作用域标识需为整数，已忽略 value=%s', value)
        return None


class MilvusVectorStore(VectorStoreBase):
    """基于 Milvus 的实现；构造时确保集合存在。"""

    backend = "milvus"
    metric_type = "COSINE"

    def __init__(
        self,
        uri: str = 'http://localhost:19530',
        token: str = "",
        collection_name: str = "",
        vector_size: int = 1024,
        db_name: str = "",
        user: str = "",
        password: str = "",
        timeout: Optional[float] = None,
    ) -> None:

        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.db_name = db_name
        self.user = user
        self.password = password
        self.timeout = timeout
        self.client = None

    def start(self) -> MilvusVectorStore:

        if MilvusClient is None:
            raise OptionalDependencyError("使用 Milvus 向量库需要 pymilvus>=2.5（含 Function 类型），请安装或升级 pymilvus") from _PYMILVUS_ERROR

        try:
            self.client = MilvusClient(
                uri=self.uri,
                token=self.token,
                db_name=self.db_name,
                user=self.user,
                password=self.password,
                timeout=self.timeout
            )
        except Exception as exc:
            raise StorageError("无法连接 Milvus：" + str(exc)) from exc

        self.create_col(self.collection_name, self.vector_size)

        return self

    def _schema(self, vector_size: int) -> Any:
        """集合 schema：稠密向量、元数据与 BM25 文本及稀疏向量一次建齐。"""
        if Function is None or FunctionType is None:
            raise OptionalDependencyError("BM25 稀疏检索需要 pymilvus>=2.5（含 Function 类型）")

        # 定义中文分析器参数
        chinese_analyzer_params = {
            "type": "chinese",
            "tokenizer": "jieba",
        }

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            # 作用域标识统一为 INT64（与主键 id 同类型），便于按租户与用户做标量过滤。
            FieldSchema(name="tenant_id", dtype=DataType.INT64),
            FieldSchema(name="user_id", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="vectors", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            # text 由 Milvus 分析器自动分词，供 BM25 使用。
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=_MAX_TEXT_LEN),
            FieldSchema(name="text_lemma", dtype=DataType.VARCHAR, max_length=_MAX_TEXT_LEN, enable_analyzer=True,analyzer_params=chinese_analyzer_params),
            # sparse 由 BM25 函数按 text 自动生成，稠密检索与关键词检索共用该集合。
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # 声明带时区的日期时间字段（使用 VARCHAR 存储 ISO 8601 字符串）
            FieldSchema(name="updated_at_tz", dtype=DataType.TIMESTAMPTZ),
            FieldSchema(name="created_at_tz", dtype=DataType.TIMESTAMPTZ),
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=True)
        schema.add_function(Function(name="text_lemma_bm25", input_field_names=["text_lemma"], output_field_names=["sparse"], function_type=FunctionType.BM25))

        return schema


    def _record(self, rid: int, vector: List[float], payload: Dict[str, Any]) -> Dict[str, Any]:
        """组装写入记录：text 优先取 text_lemmatized，回退 data 或 text。"""
        metadata = copy(payload) if payload else {}
        tenant_id = _as_scope_int(metadata.get("tenant_id"))
        user_id = _as_scope_int(metadata.get("user_id"))
        text = metadata.get("text") or ""
        text_lemma = metadata.get("text_lemma") or metadata.get("data") or metadata.get("text") or ""
        updated_at_tz = metadata.get("created_at_tz") or ""
        created_at_tz = metadata.get("created_at_tz") or ""

        metadata.pop("tenant_id", None)
        metadata.pop("user_id", None)
        metadata.pop("text", None)
        metadata.pop("text_lemma", None)
        metadata.pop("created_at_tz", None)
        metadata.pop("updated_at_tz", None)

        return {
            "id": rid,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "vectors": vector,
            "metadata": metadata,
            "text": str(text)[:_MAX_TEXT_LEN],
            "text_lemma": str(text_lemma)[:_MAX_TEXT_LEN],
            "created_at_tz": str(created_at_tz)[:64],
            "updated_at_tz": str(updated_at_tz)[:64],
        }


    def _build_param_insert(self, ids: list[int], payloads: list[dict[str, Any]], vectors: list[list[float | int]]) -> list[dict[str, Any]]:
        data = [self._record(rid, vector, payload) for rid, vector, payload in zip(ids, vectors, payloads)]
        return data

    def _build_param_search(self, filters: dict[str, Any] | None, top_k: int,
                            vectors: list[float | int]) -> dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [vectors],
            "anns_field": "vectors",
            "limit": top_k,
            "filter": self._create_filter(filters),
            "output_fields": ["*"],
        }
        return kwargs


    def _create_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """把等值过滤字典翻译为 Milvus 表达式。"""
        if not filters:
            return None
        operands: List[str] = []
        for key, value in filters.items():
            if not _SAFE_FILTER_KEY.match(key):
                raise StorageError("非法过滤键：" + str(key))
            if value == "*":
                continue

            if key == 'tenant_id' and isinstance(value, int):
                operands.append("tenant_id == " + str(value))
            elif key == 'user_id' and isinstance(value, int):
                operands.append("user_id == " + str(value))
            elif key == 'text_lemma' and isinstance(value, str):
                operands.append("text_lemma == " + str(value))
            else:
                if isinstance(value, bool):
                    operands.append("(metadata[" + json.dumps(key) + "] == " + str(value).lower() + ")")
                elif isinstance(value, (int, float)):
                    operands.append("(metadata[" + json.dumps(key) + "] == " + str(value) + ")")
                elif isinstance(value, str):
                    operands.append("(metadata[" + json.dumps(key) + "] == " + json.dumps(value) + ")")
                else:
                    raise StorageError("过滤值类型不支持：" + str(key))
        return " and ".join(operands) if operands else None

    def _parse(self, hits: List[Dict[str, Any]]) -> List[VectorStoreResult]:
        """把 Milvus 命中转成结果对象；L2 距离换算为相似度。"""
        results: List[VectorStoreResult] = []
        for value in hits:
            raw = value.get("distance")
            if raw is None:
                score = None
            elif self.metric_type == "L2":
                score = 1.0 / (1.0 + float(raw))
            else:
                score = float(raw)
            entity = value.get("entity") or {}

            entity['score'] = score
            store = VectorStoreResult.to_vector_store_result(**entity)
            results.append(store)
        return results

    def create_col(self, name: str, vector_size: int, distance: MetricType = "COSINE") -> None:
        """集合不存在时按单一 schema 创建；已存在则直接复用。"""
        if self.client.has_collection(name):
            logger.info(f"collection {name} is exist")
            return

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vectors", metric_type=distance, index_type="AUTOINDEX", index_name="vector_index")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25", index_name="sparse_index")
        self.client.create_collection(collection_name=name, schema=self._schema(vector_size), index_params=index_params)

    def _query_parse(self, rows: List[Dict[str, Any]]) -> List[VectorStoreResult]:
        results: List[VectorStoreResult] = []
        for row in rows:
            store = VectorStoreResult.to_vector_store_result(**row)
            results.append(store)
        return results

    def insert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[int]) -> None:
        """批量写入"""
        data = self._build_param_insert(ids, payloads, vectors)
        try:
            self.client.insert(collection_name=self.collection_name, data=data)
        except Exception as exc:
            raise StorageError("Milvus 写入失败：" + str(exc)) from exc


    def delete(self, id: int) -> None:
        """按 id 删除单条。"""
        self.client.delete(collection_name=self.collection_name, ids=[id])

    def update(self, id: int, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        """按 id 更新；缺省字段沿用已有记录。"""
        if vector is None or payload is None:
            existing = self.client.get(consistency_level="Strong", collection_name=self.collection_name, ids=id)
            if not existing:
                raise StorageError("待更新的向量不存在：" + str(id))
            if vector is None:
                vector = existing[0].get("vectors")
            if payload is None:
                payload = existing[0].get("metadata")
        self.client.upsert(collection_name=self.collection_name, data=self._record(id, vector, payload))

    def get(self, id: int) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""
        result = self.client.get(consistency_level="Strong",collection_name=self.collection_name, ids=id)
        if not result:
            return None
        item = result[0]
        return VectorStoreResult.to_vector_store_result(**item)



    def search(self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        """稠密向量检索；集合含稀疏字段，故显式指定 anns_field。"""
        kwargs = self._build_param_search(filters, top_k, vectors)
        try:
            hits = self.client.search(consistency_level="Strong",**kwargs)
        except Exception as exc:
            raise StorageError("Milvus 检索失败：" + str(exc)) from exc
        return self._parse(hits[0] if hits else [])


    def keyword_search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Optional[List[VectorStoreResult]]:
        """BM25 关键词检索：以原始查询文本作为稀疏向量输入。"""
        try:
            hits = self.client.search(
                consistency_level="Strong",
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse",
                limit=top_k,
                filter=self._create_filter(filters),
                output_fields=["*"],
            )
        except Exception:
            return None
        return self._parse(hits[0] if hits else [])


    def list(self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> Optional[List[VectorStoreResult]]:
        """列出集合内容；外层列表按既有约定包一层。"""
        rows = self.client.query(
            consistency_level="Strong",
            collection_name=self.collection_name,
            filter=self._create_filter(filters),
            limit=top_k or 100,
            output_fields=["*"],
        )

        results = self._query_parse(rows)
        return results


try:  # 可选依赖：异步客户端自 pymilvus 2.5 起提供，缺失时仅异步实现不可用。
    from pymilvus import AsyncMilvusClient
    _ASYNC_PYMILVUS_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover
    AsyncMilvusClient = None  # type: ignore[assignment]
    _ASYNC_PYMILVUS_ERROR = exc


class AsyncMilvusVectorStore(MilvusVectorStore):
    """Milvus 异步实现：schema、记录组装与结果解析复用同步版，只有 IO 换成 AsyncMilvusClient。

    建连与建集合都要 await，无法放进 __init__，故用 create() 类方法构造；不声明 backend，
    避免覆盖注册表里同步实现的登记。
    """
    backend = "milvus_async"
    metric_type = "COSINE"

    def __init__(
        self,
            uri: str = 'http://localhost:19530',
            token: str = "",
            collection_name: str = "",
            vector_size: int = 1024,
            db_name: str = "",
            user: str = "",
            password: str = "",
            timeout: Optional[float] = None,
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.db_name = db_name
        self.user = user
        self.password = password
        self.timeout = timeout
        self.client = None

    async def start(self) -> Any:
        """建连并确保集合存在；缺 pymilvus 或其版本过低时抛 OptionalDependencyError。"""
        if AsyncMilvusClient is None:
            raise OptionalDependencyError(
                '异步 Milvus 客户端需要 pymilvus>=2.5，请安装或升级 pymilvus'
            ) from _ASYNC_PYMILVUS_ERROR
        try:
            self.client = AsyncMilvusClient(
                uri=self.uri,
                token=self.token,
                db_name=self.db_name,
                user=self.user,
                password=self.password,
                timeout=self.timeout
            )
        except Exception as exc:
            raise StorageError('无法连接 Milvus（异步）：' + str(exc)) from exc

        await self.create_col(self.collection_name, self.vector_size)

        return self

    async def create_col(self, name: str, vector_size: int, distance: MetricType = 'COSINE') -> None:
        """集合不存在时按同一 schema 创建；已存在则直接复用。"""
        if await self.client.has_collection(name):
            logger.info('collection %s is exist', name)
            return

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name='vectors', metric_type=distance, index_type='AUTOINDEX', index_name='vector_index'
        )
        index_params.add_index(
            field_name='sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25', index_name='sparse_index'
        )
        await self.client.create_collection(
            collection_name=name, schema=self._schema(vector_size), index_params=index_params
        )

    async def insert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[int]) -> None:
        """批量写入：记录组装复用同步版。"""
        data = self._build_param_insert(ids,payloads, vectors)
        try:
            await self.client.insert(collection_name=self.collection_name, data=data)
        except Exception as exc:
            raise StorageError('Milvus 异步写入失败：' + str(exc)) from exc

    async def delete(self, id: int) -> None:
        """按 id 删除单条。"""
        await self.client.delete(collection_name=self.collection_name, ids=[id])

    async def update(
        self, id: int, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """按 id 更新；缺省字段沿用已有记录。"""
        if vector is None or payload is None:
            existing = await self.client.get(consistency_level="Strong",collection_name=self.collection_name, ids=id)
            if not existing:
                raise StorageError('待更新的向量不存在：' + str(id))
            if vector is None:
                vector = existing[0].get('vectors')
            if payload is None:
                payload = existing[0].get('metadata')
        await self.client.upsert(collection_name=self.collection_name, data=self._record(id, vector, payload))

    async def get(self, id: int) -> Optional[VectorStoreResult]:
        """按 id 读取单条，不存在返回 None。"""
        result = await self.client.get(consistency_level="Strong", collection_name=self.collection_name, ids=id)
        if not result:
            return None
        return VectorStoreResult.to_vector_store_result(**result[0])

    async def search(
        self, query: str, vectors: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[VectorStoreResult]]:
        """稠密向量检索；集合含稀疏字段，故显式指定 anns_field。"""
        kwargs = self._build_param_search(filters, top_k, vectors)
        try:
            hits = await self.client.search(consistency_level="Strong",**kwargs)
        except Exception as exc:
            raise StorageError('Milvus 异步检索失败：' + str(exc)) from exc
        return self._parse(hits[0] if hits else [])

    async def keyword_search(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[VectorStoreResult]]:
        """BM25 关键词检索：以原始查询文本作为稀疏向量输入；失败返回 None，与同步实现一致。"""
        try:
            hits = await self.client.search(
                consistency_level="Strong",
                collection_name=self.collection_name,
                data=[query],
                anns_field='sparse',
                limit=top_k,
                filter=self._create_filter(filters),
                output_fields=['*'],
            )
        except Exception:
            return None
        return self._parse(hits[0] if hits else [])

    async def list(
        self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> Optional[List[VectorStoreResult]]:
        """列出集合内容；外层列表按既有约定包一层。"""
        rows = await self.client.query(
            consistency_level="Strong",
            collection_name=self.collection_name,
            filter=self._create_filter(filters),
            limit=top_k or 100,
            output_fields=['*'],
        )
        return self._query_parse(rows or [])

