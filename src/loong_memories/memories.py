# Copyright [2026] [longgyx]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
 记忆服务统一入口：读取配置，装配 embedding、llm、rerank、db、vector、nlp 六个分区。

 三个类，只做装配，不承载业务流程：
   MemoriesBase  装配入口基类，load 与 create 走同一条流程，不直接实例化。
   Memories      同步入口，db.backend 为 sqlite、mysql、pg，产出 MemoryHandler。
   AsyncMemories 异步入口，要求异步实现类（backend 取 xxx_async），产出 AsyncMemoryHandler。

  装配细节为模块级私有函数：_load_settings、_db_kwargs、_vector_kwargs、_store_class、_build_vector、_assemble。
  两个类只保留 load 与 create 入口，不承载装配逻辑。

   from loong_memories.memories import AsyncMemories, Memories

   handler = Memories.load(yaml_path="config.yaml")        # 装载配置并装配
   handler.start()

   handler = AsyncMemories.load(yaml_path="config.yaml")   # 异步侧需 await 启动
   await handler.start()

   handler = Memories.create(settings)                    # 已持有配置时直接装配

 约束：enable 与 embedding、llm、db、vector 四个分区必须为 True；rerank 与 nlp 可选。
 关系库、向量引擎、底层向量库三层是否同步一律按实现类判定（isinstance 与协程方法），混用在装配期即报错。
 本模块顶层导入 memory.handler，需要 sqlalchemy；包根 __init__ 刻意不导出本模块。
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, cast

from loong_memories.core.config_db import DBSettings
from loong_memories.core.config_memory import MemorySettings
from loong_memories.core.config_vector import VectorSettings
from loong_memories.core.exceptions import ConfigError
from loong_memories.core.loader_config import DEFAULT_ENV_PREFIX, load_config
from loong_memories.embedding.embedder_handler import EmbedderHandler
from loong_memories.llm.llm_handler import LLMHandler
from loong_memories.memory.handler import AsyncMemoryHandler, MemoryHandler
from loong_memories.reranks.rerank_handler import RerankHandler
from loong_memories.storage.memories.registry_vector import resolve_vector_store
from loong_memories.storage.memories.vector_engine import AsyncVectorEngine, VectorEngine
from loong_memories.storage.message.db_engine import AsyncDBEngine
from loong_memories.storage.message.registry_db import resolve_db

logger = logging.getLogger(__name__)

__all__ = ["AsyncMemories", "Memories", "MemoriesBase", "REQUIRED_PARTITIONS"]

# 硬依赖分区：为 False 时直接报错；rerank 与 nlp 不在此列。
REQUIRED_PARTITIONS: Dict[str, str] = {
    "embedding": "事实向量化",
    "llm": "事实抽取",
    "db": "历史与消息审计",
    "vector": "记忆检索",
}


# 装配产物类型：同步收敛到 MemoryHandler，异步收敛到 AsyncMemoryHandler。
HandlerType = TypeVar("HandlerType", MemoryHandler, AsyncMemoryHandler)


def _load_settings(
    json_text: Optional[str],
    yaml_path: Optional[str],
    use_env: bool,
    env_prefix: str,
) -> MemorySettings:
    """两个入口共用的配置装载：YAML、JSON、环境变量依次深合并后校验。"""
    return load_config(
        json_text=json_text,
        yaml_path=yaml_path,
        use_env=use_env,
        env_prefix=env_prefix,
    )


def _db_is_async(db: Any) -> bool:
    """关系库是否异步实现：以 AsyncDBEngine 派生类为准，不看 backend 名。"""
    return isinstance(db, AsyncDBEngine)


def _engine_is_async(vector: Any) -> bool:
    """向量引擎是否异步实现：以 AsyncVectorEngine 派生类为准。"""
    return isinstance(vector, AsyncVectorEngine)


def _store_is_async(store: Any) -> bool:
    """向量库是否异步实现：看实现类的检索方法是否为协程，不看 backend 名。"""
    return inspect.iscoroutinefunction(getattr(store, "search", None))


def _check_family(
    label: str, name: str, actual: str, actual_async: bool, is_async: bool,
) -> None:
    """实现类与入口不一致时报错，报出实际类名与需要哪一族（同步或异步）。"""
    if actual_async == is_async:
        return
    need = "异步" if is_async else "同步"
    raise ConfigError(
        label + " 需要使用" + need + "的" + name + "实现类，当前为 " + actual
        + "；请改用" + need + "实现类，或切换到对应入口"
    )


def _check_settings(settings: MemorySettings, label: str) -> None:
    """校验总开关与四个硬依赖分区；实现类是否同步在装配期按实例判定。"""
    if not settings.enable:
        raise ConfigError(
            "记忆服务未启用：请设置 enable=true（或环境变量 LOONG_MEMORIES__ENABLE=true）"
        )
    for name, purpose in REQUIRED_PARTITIONS.items():
        if not getattr(settings, name).enable:
            raise ConfigError(
                label + " 必须启用 " + name + " 分区（" + purpose + "）：请设置 "
                + name + ".enable=true"
            )


def _db_kwargs(settings: DBSettings) -> Dict[str, Any]:
    """按后端组装方言构造参数：连接池与表前缀通用，其余按分区取字段。"""
    kwargs: Dict[str, Any] = settings.pool_config.model_dump()

    kwargs.update({"table_prefix":settings.table_prefix})

    # 同一方言的同步与异步后端共用前缀，用 startswith 一次覆盖两种实现。
    backend = settings.backend
    if backend.startswith("sqlite"):
        sqlite = settings.sqlite
        if not sqlite.db_path:
            raise ConfigError("未配置 SQLite 库文件路径：请设置 db.sqlite.db_path")
        kwargs.update(sqlite.model_dump())

    elif backend.startswith("mysql"):
        kwargs.update(settings.mysql.model_dump())
    else:
        kwargs.update(settings.pg.model_dump())

    return kwargs


def _vector_kwargs(settings: VectorSettings) -> Dict[str, Any]:
    """按后端组装向量库连接参数：留空字段不下传，交由实现层使用自身默认值。"""
    if settings.backend.startswith("milvus"):
        candidates = settings.milvus.model_dump()
    else:
        candidates = settings.pg.model_dump()
    return {name: value for name, value in candidates.items() if value != ""}


def _store_class(backend: str, is_async: bool) -> Any:
    """取向量库实现类：异步入口取同模块内继承同步实现的异步子类，不看 backend 名。"""
    # 配置里的 backend 可能已带 _async 后缀：先归一到基础后端名，再取同步实现类。
    base = backend[:-len('_async')] if backend.endswith('_async') else backend
    sync_cls = resolve_vector_store(base)
    if not is_async:
        return sync_cls
    module = sys.modules.get(sync_cls.__module__)
    for candidate in vars(module).values() if module is not None else ():
        if (
            isinstance(candidate, type)
            and candidate is not sync_cls
            and issubclass(candidate, sync_cls)
            and _store_is_async(candidate)
        ):
            return candidate
    raise ConfigError(
        "向量库 " + backend + " 没有可用的异步实现类：请注入 vector 或改用同步入口 Memories"
    )


def _build_vector(
    settings: VectorSettings, is_async: bool,
) -> Union[VectorEngine, AsyncVectorEngine]:
    """按实现类构造向量引擎：集合与维度统一注入，连接参数复用工厂映射表。"""
    if not settings.collection:
        raise ConfigError("未配置向量集合名：请设置 vector.collection")
    store_kwargs: Dict[str, Any] = {
        "collection_name": settings.collection,
        "vector_size": settings.vector_size,
    }
    store_kwargs.update(_vector_kwargs(settings))
    store = _store_class(str(settings.backend), is_async)(**store_kwargs)
    engine_cls = AsyncVectorEngine if is_async else VectorEngine
    return engine_cls(store, default_top_k=settings.top_k)


def _assemble(
    settings: MemorySettings,
    *,
    is_async: bool,
    handler_cls: Type[Any],
    label: str,
) -> Any:
    """校验配置后按配置装配处理器：embedding、llm、rerank、db、vector、nlp 均由配置构造。"""
    _check_settings(settings, label)

    embedder = EmbedderHandler(settings.embedding)
    llm = LLMHandler(settings.llm)
    reranker = RerankHandler(settings.rerank) if settings.rerank.enable else None
    db = resolve_db(settings.db.backend)(**_db_kwargs(settings.db))
    vector = _build_vector(settings.vector, is_async)

    # 三层都必须同族：同步引擎包住异步方言时，调用会返回未 await 的协程。
    _check_family(
        label, "关系库", type(db).__name__, _db_is_async(db), is_async,
    )
    _check_family(
        label, "向量引擎", type(vector).__name__, _engine_is_async(vector), is_async,
    )
    store = getattr(vector, "store", None)
    if store is None:
        raise ConfigError(label + " 的向量引擎缺少底层向量库实现")
    _check_family(
        label, "向量库实现", type(store).__name__, _store_is_async(store), is_async,
    )

    nlp_setting = settings.nlp if settings.nlp.enable else None
    handler = handler_cls(
        embedder=embedder,
        llm=llm,
        db=db,
        vector=vector,
        reranker=reranker,
        nlp_setting=nlp_setting,
    )
    logger.info(
        "记忆处理器已装配 mode=%s db=%s vector=%s rerank=%s nlp=%s",
        "async" if is_async else "sync",
        getattr(db, "backend", None),
        getattr(vector, "backend", None),
        reranker is not None,
        nlp_setting is not None,
    )
    return handler


class MemoriesBase(Generic[HandlerType]):
    """装配入口基类：load 与 create 共用一条装配流程，子类只声明实现族与产物类型。"""

    # 同步还是异步：决定后端族、校验方向与产物类型，子类必须覆盖。
    is_async: bool = False
    handler_cls: Type[Any] = MemoryHandler

    @classmethod
    def load(
        cls,
        json_text: Optional[str] = None,
        yaml_path: Optional[str] = None,
        use_env: bool = True,
        env_prefix: str = DEFAULT_ENV_PREFIX,
    ) -> HandlerType:
        """装载配置并装配处理器；返回对象未启动，同步侧 start()，异步侧 await start()。"""
        return cls.create(_load_settings(json_text, yaml_path, use_env, env_prefix))

    @classmethod
    def create(cls, settings: MemorySettings) -> HandlerType:
        """按已有配置装配处理器：六个分区全部由配置构造，不接受外部注入。"""
        return cast(HandlerType, _assemble(
            settings,
            is_async=cls.is_async,
            handler_cls=cls.handler_cls,
            label=cls.__name__,
        ))


class Memories(MemoriesBase[MemoryHandler]):
    """同步入口：db.backend 为 sqlite、mysql、pg，产出 MemoryHandler。"""

    is_async = False
    handler_cls = MemoryHandler


class AsyncMemories(MemoriesBase[AsyncMemoryHandler]):
    """异步入口：要求异步实现类（backend 取 xxx_async），产出 AsyncMemoryHandler。"""

    is_async = True
    handler_cls = AsyncMemoryHandler
