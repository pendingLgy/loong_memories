# loong_memories

![python](https://img.shields.io/badge/python-3.8+-3776AB)
![license](https://img.shields.io/badge/license-Apache%202.0-blue)
![status](https://img.shields.io/badge/status-dev-orange)
![tests](https://img.shields.io/badge/tests-pytest-0A9EDC)

**给大模型应用的长时记忆层**：把对话与文本抽取为结构化记忆，写入向量库与关系库，再按语义加关键词双路召回并融合排序。

loong_memories 面向需要跨会话记住用户的 LLM 应用（AI 助手、Agent、客服机器人）。它把「记忆」拆成四个稳定的动作 —— 写入、检索、更新、删除 —— 并把大模型厂商、嵌入模型、关系库、向量库全部做成可插拔后端，换一个配置键即可切换实现。

## 特性

- **四流程 API**：`add` / `search` / `update` / `delete`，入参与返回值均为 pydantic 模型，不做裸字典跨层传递。
- **事实抽取**：写入时由大模型从对话中抽取事实，提示词与 mem0 逐字一致，支持自定义 instructions。
- **双路召回**：语义检索与关键词（BM25 / 全文）召回取并集，按权重融合打分，再按阈值过滤与 top_k 截断。
- **去重与审计**：正文 MD5 指纹去重；写入、更新、删除均落一条历史记录，支持溯源。
- **可插拔后端**：注册表加工厂模式，子类声明 `backend` 即自动注册；大模型支持 OpenAI / Qwen / DeepSeek 以及任意 OpenAI 兼容服务。
- **配置三级叠加**：字段默认值、YAML 文件、JSON 字符串、环境变量按优先级深度合并，最后统一由 pydantic 校验。
- **同步与异步**：关系库与向量库均提供异步实现（配置键加 `_async` 后缀），业务层默认同步阻塞、不引入并发。
- **会话消息留存**：按 `session_scope` 只保留最近 N 条消息，为滚动压缩与长上下文对齐预留接入点。
- **ID 与作用域**：64 位雪花 ID 做主键；`user_id` / `agent_id` / `run_id` 是唯一身份来源，metadata 中的同名字段一律剥离。

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

```python
from loong_memories.core.loader_config import load_config
from loong_memories.embedding.embedder_handler import EmbedderHandler
from loong_memories.llm.llm_handler import LLMHandler
from loong_memories.memory.handler import MemoryHandler
from loong_memories.storage.memories.milvus_store import MilvusVectorStore
from loong_memories.storage.memories.vector_engine import VectorEngine
from loong_memories.storage.message.db_engine import build_database

# 1. 装载配置：字段默认值 + YAML + JSON 字符串 + 环境变量
settings = load_config(yaml_path='config.yaml')

# 2. 装配四个下层组件
db = build_database(settings.db)
db._start()

store = MilvusVectorStore(
    uri=settings.vector.milvus.uri,
    collection_name=settings.vector.collection,
    vector_size=settings.vector.vector_size,
).start()

handler = MemoryHandler(
    settings,
    embedder=EmbedderHandler(settings.embedding),
    llm=LLMHandler(settings.llm),
    db=db,
    vector=VectorEngine(store, default_top_k=settings.vector.top_k),
)

# 3. 写入：大模型抽取事实，去重后落向量库与关系库
result = handler.add(
    [{"role": "user", "content": "我是一名后端工程师，最近在写长时记忆服务"},
     {"role": "assistant", "content": "记下了。"}],
    user_id="u_1001",
)

# 4. 检索：语义与关键词双路召回，融合打分后返回
hits = handler.search("我的职业是什么", filters={"user_id": "u_1001"}, top_k=5)
for item in hits.results:
    print(item.id, item.score, item.memory)
```


## 配置

### 装载入口

```python
from loong_memories.core.loader_config import load_config, load_config_from_env
from loong_memories.core.loader_config import load_config_from_json, load_config_from_yaml

settings = load_config(json_text=None, yaml_path=None, use_env=True)
```

优先级从低到高：字段默认值、YAML 文件、JSON 字符串、环境变量。同名键深度合并，最终统一由 `MemorySettings` 校验；任何解析或校验失败都抛 `ConfigError`。

`MemorySettings` 由四个分区组成，均设置 `extra=forbid`，字段名拼错会直接报错：

| 分区 | 模型 | 关键字段 |
| --- | --- | --- |
| `embedding` | `EmbeddingSettings` | `provider`、`model`、`dim`、`batch_size`、`api_key`、`base_url`、`timeout` |
| `llm` | `LLMSettings` | `provider`、`model`、`temperature`、`max_tokens`、`api_key`、`base_url`、`timeout` |
| `db` | `DBSettings` | `backend`、`table_prefix`、`sqlite` / `mysql` / `pg`、`pool_config` |
| `vector` | `VectorSettings` | `backend`、`collection`、`vector_size`、`top_k`、`metric_type`、`milvus` / `pg` |

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
db:
  enable: true
  backend: sqlite
  table_prefix: lm_
  sqlite:
    path: data/memory.db
vector:
  enable: true
  backend: milvus
  collection: lm_memories
  vector_size: 1024
  top_k: 8
  metric_type: COSINE
  milvus:
    uri: http://localhost:19530
```

### 环境变量

前缀固定为 `LOONG_MEMORIES__`，层级用双下划线连接，字段名不区分大小写：

| 环境变量 | 映射字段 |
| --- | --- |
| `LOONG_MEMORIES__LLM__PROVIDER` | `llm.provider` |
| `LOONG_MEMORIES__EMBEDDING__DIM` | `embedding.dim` |
| `LOONG_MEMORIES__DB__BACKEND` | `db.backend` |
| `LOONG_MEMORIES__DB__SQLITE__PATH` | `db.sqlite.path` |
| `LOONG_MEMORIES__DB__POOL_CONFIG__POOL_SIZE` | `db.pool_config.pool_size` |
| `LOONG_MEMORIES__VECTOR__MILVUS__URI` | `vector.milvus.uri` |

取值默认保持字符串，交由 pydantic 转换类型；仅当值以 `[` 或 `{` 开头时才按 JSON 解析，用于表达列表与嵌套对象（`0123` 这类前导零口令不会被误判为 JSON）。

API Key 也可以留空，由各 provider 按约定读取 `OPENAI_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`。


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

### 新增模型厂商

OpenAI 兼容服务直接继承现有实现，只覆盖默认模型、接口地址、Key 环境变量与批量上限：

```python
from loong_memories.llm.openai_llm import OpenAILLMProvider
from loong_memories.embedding.openai_embedder import OpenAIEmbedder

class MyLLM(OpenAILLMProvider):
    backend = "my_llm"
    default_model = "my-chat-model"
```


## 路线图

- 实现顶层门面 `loong_memories.py` 与包级导出，收敛当前的手工装配流程。
- 对话历史滚动压缩与长文本记忆对齐：入口落在 `memory.handler`，提示词落在 `memory.prompts`，阈值配置落在 `core.config_memory`。
- 向量分区放行异步后端，补齐异步编排。
- 提供开箱可用的示例工程与 Docker Compose 依赖编排。

## 许可

Apache 2.0，作者 loong（longgyx@gmail.com）。

## 开发贡献

开发环境与依赖管理、命令分类、代码检查、测试与集成测试配置、目录结构与测试清单、待实现项见 [DEVELOPMENT.md](DEVELOPMENT.md)。
