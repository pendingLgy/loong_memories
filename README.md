# loong_memories

![python](https://img.shields.io/badge/python-3.8+-3776AB)
![license](https://img.shields.io/badge/license-Apache%202.0-blue)
![status](https://img.shields.io/badge/status-dev-orange)
![tests](https://img.shields.io/badge/tests-pytest-0A9EDC)

**给大模型应用的长时记忆层**：把对话与文本抽取为结构化记忆，写入向量库与关系库，再按语义加关键词双路召回并融合排序。

loong_memories 面向需要跨会话记住用户的 LLM 应用（AI 助手、Agent、客服机器人）。它把「记忆」拆成六个稳定动作 —— 读取、分页、写入、检索、更新、删除 —— 并把大模型厂商、嵌入模型、关系库、向量库全部做成可插拔后端，换一个配置键即可切换实现。

> 顶层门面 `loong_memories.memories` 提供 `Memories` 与 `AsyncMemories` 两个入口，`load` / `create` 一次完成配置装载与装配；业务调用仍落在 `MemoryHandler` 与 `AsyncMemoryHandler` 上。

## 特性

- **六流程 API**：`get` / `page_query` / `add` / `search` / `update` / `delete`，入参与返回值均为 pydantic 模型，不做裸字典跨层传递。
- **租户硬隔离**：`tenant_id` 是每个流程的必填位置参数，`user_id` 可选收窄；两者走存储层独立列，不再混进 `filter` 条件。
- **事实抽取**：写入时由大模型从对话中抽取事实，提示词与 mem0 逐字一致，支持自定义 instructions。
- **双路召回**：语义检索与关键词（BM25 / 全文）召回取并集，按权重融合打分，再按阈值过滤与 `top_k` 截断。
- **去重与审计**：正文 MD5 指纹去重；写入、更新、删除各落一条历史记录，支持溯源。
- **可插拔后端**：注册表加工厂模式，子类声明 `backend` 即自动注册；大模型支持 OpenAI / Qwen / DeepSeek 以及任意 OpenAI 兼容服务。
- **配置三级叠加**：字段默认值、YAML 文件、JSON 字符串、环境变量按优先级深度合并，最后统一用 pydantic 校验。
- **同步与异步**：关系库与向量库均提供异步实现（配置键为 `_async` 后缀），入口按实现类分族并在装配期校验，业务层默认同步阻塞、不引入并发。
- **会话消息留存**：按 `session_scope` 只保留最近 N 条消息，为滚动压缩与长上下文对齐预留接入点。
- **统一入口**：`Memories` / `AsyncMemories` 的 `load` 与 `create` 一次完成配置装载、六分区构造与三层实现族校验，装配失败在启动前抛 `ConfigError`。
- **ID 与作用域**：64 位雪花 ID 作主键；`agent_id` 是唯一的会话分组来源，metadata 中的身份字段一律剥离。

## 安装

```bash
pip install loong_memories                     # 基础依赖（openai、pydantic、PyYAML）
pip install loong_memories[sqlalchemy]         # SQLite 关系库（SQLAlchemy Core，通用实现的前提）
pip install loong_memories[milvus]             # Milvus 向量库
pip install loong_memories[pg]                 # PostgreSQL 关系库与 pgvector 向量库
pip install loong_memories[mysql]              # MySQL 关系库
pip install loong_memories[parsers]            # spaCy 与中文模型（BM25 词形还原）
```

| 可选依赖 | 解锁的能力 |
| --- | --- |
| `sqlalchemy` | 全部关系库后端；`storage.message.db_engine` 顶层即导入 SQLAlchemy，未安装时记忆层无法导入 |
| `pymilvus`（建议 2.5 及以上） | Milvus 向量库，含 BM25 Function 与 `AsyncMilvusClient` |
| `psycopg[binary]` / `pgvector` / `asyncpg` | PostgreSQL 关系库与 pgvector 向量库 |
| `pymysql` / `aiomysql` | MySQL 同步与异步驱动 |
| `aiosqlite` | SQLite 异步驱动 |
| `spacy` 加中文模型 | 关键词检索的词形还原；缺失时降级为原文 |

## 快速开始

### 装配

顶层门面 `memories.py` 已经可用，装配只需一次调用：`Memories.load` 装载配置并构造同步处理器，`AsyncMemories.load` 构造异步处理器。两个类都提供 `load` 与 `create`，`load` 读配置，`create` 直接吃已构造好的 `MemorySettings`。

```python
from loong_memories.memories import AsyncMemories, Memories

handler = Memories.load(yaml_path='config.yaml')   # 或 Memories.create(settings)
handler.start()                                        # 启动关系库与向量库连接

async_handler = AsyncMemories.load(yaml_path='config.yaml')
await async_handler.start()                            # 异步侧必须 await
```

`load` 的签名是 `(json_text=None, yaml_path=None, use_env=True, env_prefix=LOONG_MEMORIES__)`，与 `load_config` 一致；它**不接收** `embedder` / `llm` / `reranker` / `db` / `vector` 注入，六个分区全部由配置构造。需要替换后端时改配置，或在测试里直接构造 `MemoryHandler`。

装配期会做三轮硬校验，任一不过直接抛 `ConfigError`：

1. 总开关 `enable` 与四个硬依赖分区 `embedding` / `llm` / `db` / `vector` 必须为 `True`，`rerank` 与 `nlp` 可选；
2. `db.sqlite.db_path`、`vector.collection` 这类必填项不能为空；
3. 关系库、向量引擎、底层向量库三层的**实现族**必须与入口一致 —— 全部按实现类判定（`isinstance` 与协程方法），不看 `backend` 名字。

第三条是关键：`Memories` 只接受同步实现类，`AsyncMemories` 只接受异步实现类。混用（例如同步入口配了 `db.backend=sqlite_async`，或同步引擎里包了异步向量库）在装配期就报错，不会拖到运行期拿到未 `await` 的协程。


### 写入

```python
result = handler.add(
    [{"role": "user", "content": "我是一名后端工程师，最近在写长时记忆服务"},
     {"role": "assistant", "content": "记下了。"}],
    tenant_id=7001,          # 必填：租户隔离边界
    user_id=9527,            # 可选：收窄到具体用户
    metadata={"source": "chat"},
)

for item in result.results:
    print(item.id, item.memory, item.hash)
```

`add` 内部顺序：召回既有记忆（既作抽取上下文，也作指纹去重）→ 大模型抽取事实 → 指纹去重 → 批量嵌入 → 写向量库 → 写历史与消息。

### 检索

```python
hits = handler.search(
    "我的职业是什么",
    tenant_id=7001,
    user_id=9527,
    top_k=5,
    threshold=0.1,
    filter={"source": "chat"},   # 关键字参数
    rerank=True,
    explain=True,
)

for item in hits.results:
    print(item.id, item.score, item.memory)
    if item.score_details:
        print(item.score_details)
```

`filter` 支持 `eq` / `ne` / `gt` / `gte` / `lt` / `lte` / `in` / `nin` / `contains` / `icontains` 以及 `*` 存在性判断；`metadata` 上的键走 JSONB 包含查询。

### 读取与分页

```python
one = handler.get(memory_id, tenant_id=7001, user_id=9527)
if one.result is not None:              # GetResult.result 是单条，不是列表
    print(one.result.memory, one.result.updated_at_tz)

page = handler.page_query(
    tenant_id=7001,
    user_id=9527,
    page_num=0,
    page_size=20,
    filter={"source": "chat"},
)
print(page.total_size, len(page.results), page.page_num, page.page_size)
```

`page_query` 直接复用向量库的 `list` 抓取 `(page_num + 1) * page_size` 条再切片；存储层没有独立计数接口，`total_size` 表示本次抓取窗口内的命中条数。

### 更新与删除

```python
upd = handler.update(memory_id, tenant_id=7001, user_id=9527, text="新的正文")
print(upd.result.memory)

dele = handler.delete(memory_id, tenant_id=7001, user_id=9527)
print(dele.id)
```

`update` 只有在正文真正变化时才重算向量并落库；正文与旧值一致时视为无变化，直接返回当前条目、不写库。目标记忆不存在时抛 `DataError`。


## API 参考

### 方法签名

同步与异步两侧签名完全一致，异步版只在使用时加 `await`。

| 方法 | 签名 | 返回 |
| --- | --- | --- |
| `get` | `(memory_id, tenant_id, user_id=None)` | `GetResult` |
| `page_query` | `(tenant_id, user_id=None, page_num=0, page_size=10, *, filter=None)` | `PageQueryResult` |
| `add` | `(messages, tenant_id, user_id=None, *, agent_id=None, metadata=None)` | `AddResult` |
| `search` | `(query, tenant_id, user_id=None, *, filter=None, top_k=None, threshold=None, rerank=False, explain=False)` | `SearchResult` |
| `update` | `(memory_id, tenant_id, user_id=None, *, text=None, metadata=None)` | `UpdateResult` |
| `delete` | `(memory_id, tenant_id, user_id=None)` | `DeleteResult` |
| `start` | `()` | `None` |

要点：

- `tenant_id` 与 `user_id` 是位置参数，可用关键字形式传；`filter` / `top_k` / `threshold` / `rerank` / `explain` 与 `text` / `metadata` 为关键字专用参数。
- `filter` 的键名是单数，不是 `filters`。
- 异步版多一个 `async def start()`，装配后需要 `await handler.start()`。

### 返回模型

| 模型 | 关键字段 |
| --- | --- |
| `MemoryItem` | `id`、`memory`、`score`、`hash`、`created_at_tz`、`updated_at_tz`、`tenant_id`、`user_id`、`agent_id`、`actor_id`、`role`、`metadata`、`score_details` |
| `AddResult` | `results: List[MemoryItem]` |
| `GetResult` | `result: Optional[MemoryItem]` |
| `PageQueryResult` | `page_num`、`page_size`、`total_size`、`results: List[MemoryItem]` |
| `SearchResult` | `results: List[MemoryItem]`，按融合分数降序 |
| `UpdateResult` | `result: Optional[MemoryItem]` |
| `DeleteResult` | `id: int` |

### 默认参数

| 常量 | 值 | 用途 |
| --- | ---: | --- |
| `DEFAULT_THRESHOLD` | 0.1 | 检索相似度下限 |
| `DEFAULT_TOP_K` | 20 | `search` 默认返回条数 |
| `DEFAULT_CONTEXT_TOP_K` | 10 | 写入前召回既有记忆的条数 |
| `DEFAULT_KEYWORD_WEIGHT` | 0.5 | 关键词路径在融合打分中的权重 |
| `DEFAULT_HISTORY_MESSAGES` | 10 | 提示词里携带的最近对话条数 |

语义路内部召回窗口为 `max(top_k * 4, 60)`，关键词路复用同一窗口，避免融合前后候选集被截断两次。


## 配置

本节给出常用字段、装载入口与示例；**完整的字段清单**（类型、默认值、约束、环境变量名、校验报错与扩展指引）统一维护在 [CONFIGURATION.md](CONFIGURATION.md)，改配置前先对照该表。

### 装载入口

```python
from loong_memories.core.loader_config import load_config
from loong_memories.core.loader_config import load_config_from_env
from loong_memories.core.loader_config import load_config_from_json, load_config_from_yaml

settings = load_config(json_text=None, yaml_path=None, use_env=True)
```

优先级从低到高：字段默认值、YAML 文件、JSON 字符串、环境变量。同名键深度合并，最终统一用 `MemorySettings` 校验；任何解析或校验失败都抛 `ConfigError`。

`MemorySettings` 由六个分区组成（顶层另有总开关 `enable`），全部设置 `extra=forbid`，字段名拼错会直接校验失败：

| 分区 | 模型 | 关键字段 |
| --- | --- | --- |
| `embedding` | `EmbeddingSettings` | `provider`、`model`、`dim`、`batch_size`、`api_key`、`base_url`、`timeout` |
| `llm` | `LLMSettings` | `provider`、`model`、`temperature`、`max_tokens`、`api_key`、`base_url`、`timeout` |
| `rerank` | `RerankSettings` | `provider`、`model`、`top_k`、`max_documents`、`max_tokens`、`api_key`、`base_url`、`timeout` |
| `db` | `DBSettings` | `backend`、`table_prefix`、`sqlite` / `mysql` / `pg`、`pool_config` |
| `vector` | `VectorSettings` | `backend`、`collection`、`vector_size`、`top_k`、`metric_type`、`milvus` / `pg` |
| `nlp` | `NlpSettings` | `enable`、`nlp_name`（默认 `zh_core_web_md`，需 parsers 依赖） |

后端可选值：`db.backend` 取 `sqlite`、`mysql`、`pg`、`sqlite_async`、`mysql_async`、`pg_async`；`vector.backend` 取 `milvus`、`pg`、`milvus_async`、`pg_async`。同步入口配无后缀值，异步入口配 `_async` 值，混用在装配期报错。

`MemorySettings` 由六个分区组成（顶层还有总开关 `enable`），每个子模型都是 `extra=forbid`，字段名拼错会直接校验失败；另校验 `embedding.dim` 与 `vector.vector_size` 是否一致，不一致直接拒绝启动。

配置字段的类型、默认值、约束与环境变量对照见 [CONFIGURATION.md](CONFIGURATION.md)。

### YAML 示例

```yaml
embedding:
  enable: true
  provider: openai
  model: text-embedding-3-small
  dim: 1024
  batch_size: 64
llm:
  enable: true
  provider: deepseek
  model: deepseek-chat
  temperature: 0.3
rerank:
  enable: true
  provider: qwen
  model: gte-rerank
  top_k: 5
db:
  enable: true
  backend: sqlite
  table_prefix: lm_
  sqlite:
    db_path: data/memory.db
vector:
  enable: true
  backend: milvus
  collection: lm_memories
  vector_size: 1024
  top_k: 8
  metric_type: COSINE
  milvus:
    uri: http://localhost:19530
nlp:
  enable: true
  nlp_name: zh_core_web_md
```


### 环境变量

前缀固定为 `LOONG_MEMORIES__`，层级用双下划线连接，字段名不区分大小写：

| 环境变量 | 映射字段 |
| --- | --- |
| `LOONG_MEMORIES__LLM__PROVIDER` | `llm.provider` |
| `LOONG_MEMORIES__EMBEDDING__DIM` | `embedding.dim` |
| `LOONG_MEMORIES__DB__BACKEND` | `db.backend` |
| `LOONG_MEMORIES__DB__SQLITE__DB_PATH` | `db.sqlite.db_path` |
| `LOONG_MEMORIES__DB__POOL_CONFIG__POOL_SIZE` | `db.pool_config.pool_size` |
| `LOONG_MEMORIES__VECTOR__MILVUS__URI` | `vector.milvus.uri` |

取值默认保持字符串，交由 pydantic 转换类型；仅当值以 `[` 或 `{` 开头时才按 JSON 解析，用于表达列表与嵌套对象。

API Key 也可以留空，由各 provider 按约定读取 `OPENAI_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`。

每个字段对应的环境变量名可按「前缀 + 层级双下划线 + 字段名大写」推导，全量对照见 [CONFIGURATION.md](CONFIGURATION.md) 的各分区字段表。


## 数据流

```text
add          messages → 召回既有记忆（嵌入 + 双路召回）→ 大模型抽取事实
                   → 指纹去重 → 批量嵌入 → 写向量库 → 写历史与消息

search       query → 嵌入 → 双路召回 + 融合打分 → 阈值过滤与截断 → 可选重排

get          memory_id + tenant_id（+ user_id）→ 向量库读取 → 组装 MemoryItem

page_query   tenant_id（+ user_id）→ 向量库 list 抓取窗口 → 按页切片

update       get → 合并 payload → 正文变化才重算向量 → 写向量库 + 历史 UPDATE

delete       get → 向量库物理删除 + 历史 DELETE
```

写入时 payload 的核心键为 `text`、`text_lemma`、`hash`、`created_at_tz`、`updated_at_tz`；租户与用户由显式入参提供，`metadata` 里的同名键一律剥离。

| 阶段 | 关键校验 | 失败策略 |
| --- | --- | --- |
| 消息归一 | 必须是字符串、字典或字典列表；每条必须有 `content` | 抛 `DataError` |
| 嵌入 | 逐条比对 `embedding.dim` | 抛 `EmbeddingError` |
| 事实抽取 | 模型响应按 JSON 解析，兼容代码块包裹 | 抛 `LLMError`；非 JSON 视为未抽取 |
| 向量写入 | 维度与批量长度校验 | 抛 `StorageError` |
| 历史与消息 | 无 | 只告警，不回滚主写入 |
| 重排 | 文档数不超过 `max_documents` | 异常降级为原顺序，给 0.5 中立分 |

## 扩展点

### 新增存储后端

继承抽象基类并声明 `backend`，注册与配置切换自动生效：

```python
from loong_memories.storage.message.base import BaseDB
from loong_memories.storage.memories.base import VectorStoreBase

class MyDB(BaseDB):
    backend = "my_db"          # 声明即自动注册，env 里 db.backend=my_db 即可切换
    ...

class MyVectorStore(VectorStoreBase):
    backend = "my_vs"
    metric_type = "COSINE"
    ...
```

向量库需遵守契约：`search` 返回的 `score` 越大越相似。使用距离度量的实现要先转相似度，例如 COSINE 用 `max(0.0, 1.0 - distance)`。

注意：关系库与向量库的实例化已上收到门面 `memories.py`（`_db_kwargs`、`_vector_kwargs`、`_store_class`），`db_engine` 里的 `build_database` 与 `vector_engine` 里的 `build_vector_engine`、`_backend_kwargs` 均已移除。新增后端时仍需同步 `core.config_db` 或 `core.config_vector` 的 `Literal`，并在 `vector_engine.build_vector_store` 与两个注册表补分支；门面先把 `backend` 归一，再经注册表解析实现类。

### 新增模型厂商

OpenAI 兼容服务直接继承现有实现，只需覆盖默认模型、接口地址、Key 环境变量与批量上限：

```python
from loong_memories.llm.openai_llm import OpenAILLMProvider
from loong_memories.embedding.openai_embedder import OpenAIEmbedder

class MyLLM(OpenAILLMProvider):
    backend = "my_llm"
    default_model = "my-chat-model"
```

### 新增返回字段

`MemoryItem` 与 `VectorStoreResult` 都设置了 `extra=ignore`，新增字段需要同时改模型定义与 `_to_item` 的组装逻辑，否则会被静默丢弃。

## 路线图

- 已实现顶层门面 `memories.py`（`Memories` 与 `AsyncMemories`）；剩余工作是包级导出与示例工程。
- 对话历史滚动压缩与长文本记忆对齐：入口落在 `memory.handler`，提示词落在 `memory.prompts`，阈值配置落在 `core.config_memory`。
- 向量分区已放开异步后端配置键（`milvus_async`、`pg_async`），异步编排由 `AsyncMemories` 承接。
- 提供开箱可用的示例工程与 Docker Compose 依赖编排。

## 许可

Apache 2.0，作者 loong（longgyx@gmail.com）。

## 开发贡献

开发环境与依赖管理、命令分类、代码检查、测试与集成测试配置、目录结构与测试清单、待实现项详见 [DEVELOPMENT.md](DEVELOPMENT.md)。



