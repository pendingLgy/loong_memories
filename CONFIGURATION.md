# loong_memories 配置参考

> 本文按 `src/loong_memories/core/config_*.py` 的字段声明与 `loader_config.py` 的装载逻辑实测整理，字段、默认值、约束与源码一一对应。
> 全部分区模型均为 `extra="forbid"`：字段名拼错会直接校验失败并抛 `ConfigError`，不会静默忽略。

## 1. 配置从哪里来

| 入口 | 签名 | 作用 |
| --- | --- | --- |
| `load_config` | `(json_text=None, yaml_path=None, use_env=True, env_prefix=LOONG_MEMORIES__)` | 总入口，按优先级合并三个来源后交给 `MemorySettings` 校验 |
| `load_config_from_yaml` | `(path)` | 读取 YAML 文件，空文件返回空字典，路径不存在抛 `ConfigError` |
| `load_config_from_json` | `(text)` | 解析 JSON 字符串，顶层必须是对象 |
| `load_config_from_env` | `(prefix=LOONG_MEMORIES__)` | 扫描进程环境变量，把前缀匹配的键转换为嵌套字典 |

优先级从低到高：**字段默认值 -> YAML 文件 -> JSON 字符串 -> 环境变量**。同名键深度合并，两边都是字典则递归，否则整体覆盖。解析或校验失败统一抛 `ConfigError`，便于上层单点捕获。

模块级常量：

| 常量 | 值 | 作用 |
| --- | --- | --- |
| `DEFAULT_ENV_PREFIX` | `LOONG_MEMORIES__` | 环境变量前缀，以双下划线结尾 |
| `ENV_NESTED_DELIMITER` | `__` | 环境变量名中的层级切分符 |

### 1.1 环境变量规则

- 变量名 = `LOONG_MEMORIES__` + 各层字段名大写 + `__` 分隔。例如 `LOONG_MEMORIES__DB__SQLITE__DB_PATH` 映射到 `db.sqlite.db_path`。
- 字段名不区分大小写，解析时统一转小写。
- 取值默认保持**字符串**，由 pydantic 转换类型；仅当值以 `[` 或 `{` 开头时按 JSON 解析，用于表达列表与嵌套对象，JSON 解析失败会记 warning 并回退为字符串。
- 缺少字段名的前缀键、无法解析层级的键记 warning 后跳过，不中断装载。
- `MemorySettings` 主动关闭了 pydantic-settings 的原生 env、.env 与 secrets 来源（`settings_customise_sources` 只返回 `init_settings`），环境变量只有 `loader_config` 一个入口，`use_env=False` 才有确定语义。

## 2. 顶层：MemorySettings

源文件 `core/config_memory.py`。`MemorySettings` 是唯一交给 `load_config` 校验的根模型，其余分区都是它的嵌套字段。

| 字段 | 类型 | 默认值 | 环境变量 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | `LOONG_MEMORIES__ENABLE` | 记忆服务总开关。为 False 时门面不构造任何记忆组件，各分区的 `enable` 也不再生效 |
| `embedding` | EmbeddingSettings | 默认实例 | 见下文 | 向量化分区，供嵌入器使用 |
| `llm` | LLMSettings | 默认实例 | 见下文 | 大模型分区，供事实抽取与摘要使用 |
| `rerank` | RerankSettings | 默认实例 | 见下文 | 重排分区，供检索结果重排使用 |
| `db` | DBSettings | 默认实例 | 见下文 | 关系库分区，承载历史与消息 |
| `vector` | VectorSettings | 默认实例 | 见下文 | 向量库分区，承载记忆检索 |
| `nlp` | NlpSettings | 默认实例 | 见下文 | 自然语言处理分区，供 BM25 词形还原使用 |

### 2.1 顶层校验

| 校验 | 规则 | 失败表现 |
| --- | --- | --- |
| 字段封闭 | `model_config = SettingsConfigDict(extra="forbid")` | 出现未知键直接抛错 |
| 维度一致 | 仅当 `embedding.enable` 与 `vector.enable` **同时为 True** 时校验 `embedding.dim == vector.vector_size` | 不一致抛 `ValueError`，经 `load_config` 包装为 `ConfigError` |
| 来源收敛 | `settings_customise_sources` 只保留 `init_settings` | 直接 `MemorySettings()` 时不会读取进程环境变量 |

## 3. 分区速查

| 分区 | 模型 | 源文件 | 门面是否强制启用 |
| --- | --- | --- | --- |
| `embedding` | `EmbeddingSettings` | `config_llm.py` | 必需 |
| `llm` | `LLMSettings` | `config_llm.py` | 必需 |
| `rerank` | `RerankSettings` | `config_llm.py` | 可选 |
| `db` | `DBSettings` | `config_db.py` | 必需 |
| `vector` | `VectorSettings` | `config_vector.py` | 必需 |
| `nlp` | `NlpSettings` | `config_nlp.py` | 可选 |

「必需」指顶层 `enable=True` 时该分区的 `enable` 也必须为 True，否则 `Memories.load` 在装配期抛 `ConfigError`。必需集合对应门面的 `REQUIRED_PARTITIONS`（embedding、llm、db、vector）。

## 4. embedding 分区（EmbeddingSettings）

源文件 `config_llm.py`。供 `EmbedderHandler` 使用，把事实文本转成向量。

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | - | 是否启用该分区；为 False 时门面不初始化嵌入器 |
| `provider` | str | `openai` | - | 嵌入模型注册表键。内置 `openai`、`qwen`，继承 `BaseEmbeddingProvider` 即自动注册 |
| `model` | str | 空串 | - | 模型名。留空时使用 provider 的 `default_model` |
| `batch_size` | int | `128` | `gt=0` | 单次请求的最大文本条数；provider 自身有更小上限时以其为准（qwen 为 10） |
| `dim` | int | `1024` | `gt=0` | 期望输出维度，向量化结果逐条按此校验；必须与 `vector.vector_size` 一致 |
| `api_key` | str | 空串 | - | 留空时读取 provider 约定的环境变量（`OPENAI_API_KEY`、`DASHSCOPE_API_KEY`） |
| `base_url` | str | 空串 | - | 留空时用 provider 默认地址，可指向自建网关或代理 |
| `timeout` | float | `60.0` | `gt=0` | 单次嵌入请求超时（秒）。0 或负数会让 HTTP 客户端立即超时 |

环境变量示例：

```bash
LOONG_MEMORIES__EMBEDDING__ENABLE=true
LOONG_MEMORIES__EMBEDDING__PROVIDER=qwen
LOONG_MEMORIES__EMBEDDING__MODEL=text-embedding-v3
LOONG_MEMORIES__EMBEDDING__DIM=1024
LOONG_MEMORIES__EMBEDDING__BATCH_SIZE=64
```

## 5. llm 分区（LLMSettings）

源文件 `config_llm.py`。供 `LLMHandler` 使用，负责写入时的事实抽取与后续的历史摘要。

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | - | 是否启用该分区；为 False 时门面不初始化大模型 |
| `provider` | str | `openai` | - | 大模型注册表键。内置 `openai`、`qwen`、`deepseek`，后两者兼容 OpenAI 协议 |
| `model` | str | 空串 | - | 模型名。留空时使用 provider 的 `default_model` |
| `temperature` | float | `0.7` | `ge=0.0`、`le=2.0` | 采样温度，超出 0 到 2 直接校验失败 |
| `max_tokens` | int | `0` | `ge=0` | 单次生成上限。`0` 表示不显式传入，由服务端决定 |
| `api_key` | str | 空串 | - | 留空时读取 `OPENAI_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY` |
| `base_url` | str | 空串 | - | 留空时用 provider 默认地址 |
| `timeout` | float | `60.0` | `gt=0` | 单次对话请求超时（秒） |

环境变量示例：

```bash
LOONG_MEMORIES__LLM__ENABLE=true
LOONG_MEMORIES__LLM__PROVIDER=deepseek
LOONG_MEMORIES__LLM__MODEL=deepseek-chat
LOONG_MEMORIES__LLM__TEMPERATURE=0.3
```

## 6. rerank 分区（RerankSettings）

源文件 `config_llm.py`。供 `RerankHandler` 使用，对召回结果做二次重排。分区可选，`enable=False` 时检索直接用融合打分结果。

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | - | 是否启用重排；为 False 时门面不构造 reranker |
| `provider` | str | `qwen` | - | 重排注册表键。内置 `qwen`（DashScope text-rerank） |
| `model` | str | 空串 | - | 模型名。留空时使用 provider 的 `default_model` |
| `top_k` | int | `5` | `gt=0` | 默认返回条数；调用方显式传 `top_k` 时以调用方为准 |
| `max_tokens` | int | `0` | `ge=0` | 兼容保留字段：DashScope text-rerank 接口不消费，当前配置不生效 |
| `max_documents` | int | `100` | `gt=0` | 单次请求参与重排的最大文档数，超出部分截断，避免请求体过大 |
| `api_key` | str | 空串 | - | 留空时读取 `DASHSCOPE_API_KEY` |
| `base_url` | str | 空串 | - | 留空时用 provider 默认地址 |
| `timeout` | float | `60.0` | `gt=0` | 单次重排请求超时（秒） |

异常降级：provider 调用失败时按原顺序返回文档并给 0.5 中立分，不阻断检索主链路。

环境变量示例：

```bash
LOONG_MEMORIES__RERANK__ENABLE=true
LOONG_MEMORIES__RERANK__PROVIDER=qwen
LOONG_MEMORIES__RERANK__MODEL=gte-rerank
LOONG_MEMORIES__RERANK__MAX_DOCUMENTS=100
```

## 7. nlp 分区（NlpSettings）

源文件 `config_nlp.py`。控制 BM25 关键词是否走词形还原；分区可选，关闭时关键词检索直接用原文。

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | - | 是否启用 NLP 处理；为 False 时不做词形还原，`supports_keyword` 判定为假 |
| `nlp_name` | str | `zh_core_web_md` | - | spaCy 模型名，需安装 `parsers` 依赖组与对应中文模型 |

环境变量示例：

```bash
LOONG_MEMORIES__NLP__ENABLE=true
LOONG_MEMORIES__NLP__NLP_NAME=zh_core_web_md
```

## 8. db 分区（DBSettings）

源文件 `config_db.py`。承载历史入库与最近消息留存，按后端名分派到 `sqlite`、`mysql`、`pg` 三个子分区的连接参数。

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | - | 是否启用关系库；为 False 时门面不初始化 |
| `backend` | Literal | `sqlite` | 见下 | 后端名，对应关系库注册表键 |
| `sqlite` | SQLiteSettings | 默认实例 | - | SQLite 连接参数 |
| `mysql` | MySQLSettings | 默认实例 | - | MySQL 连接参数 |
| `pg` | PGSettings | 默认实例 | - | PostgreSQL 连接参数 |
| `table_prefix` | str | 空串 | - | 表名前缀，用于同库多套实例共存 |
| `pool_config` | DBPoolSettings | 默认实例 | - | 连接池参数，db 与 vector 共用同一模型 |

`backend` 可选值（六个）：

| 取值 | 实现族 | 说明 |
| --- | --- | --- |
| `sqlite` | 同步 | SQLiteDB，默认可跑，适合本地开发 |
| `mysql` | 同步 | MySQLDB，需 `mysql` 依赖组与实例 |
| `pg` | 同步 | PGDB，需 `pg` 依赖组与实例 |
| `sqlite_async` | 异步 | AsyncSQLiteDB，需 `aiosqlite` |
| `mysql_async` | 异步 | AsyncMySQLDB，需 `aiomysql` 或 `asyncmy` |
| `pg_async` | 异步 | AsyncPGDB，需 `asyncpg` |

同步入口配无后缀值，异步入口配 `_async` 值；混用在装配期抛 `ConfigError`（关系库按实现类判定，向量库先按 `_async` 后缀归一后取协程子类）。

### 8.1 sqlite（SQLiteSettings）

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `db_path` | str | 空串 | 运行时必填 | 库文件路径。留空时门面抛 `ConfigError`（`_db_kwargs` 校验），不会回退到临时库 |
| `timeout` | float | `30.0` | - | 写锁等待超时（秒），避免并发写入直接抛 database is locked |
| `isolation_level` | str | 空串 | - | 事务隔离级别：`DEFERRED`、`IMMEDIATE`、`EXCLUSIVE`、`AUTOCOMMIT`；留空用驱动默认 |
| `check_same_thread` | bool | `False` | - | 是否强制检查同线程访问；多线程与连接池场景保持 False |

注意字段名是 `db_path`，不是 `path`；环境变量为 `LOONG_MEMORIES__DB__SQLITE__DB_PATH`。

### 8.2 mysql（MySQLSettings）

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `host` | str | `localhost` | - | 主机 |
| `port` | int | `3306` | `gt=0` | 端口 |
| `user` | str | `root` | - | 用户名 |
| `password` | str | 空串 | - | 密码 |
| `database` | str | 空串 | - | 库名 |
| `charset` | str | `utf8mb4` | - | 字符集 |
| `connect_timeout` | int | `10` | - | 建连超时（秒） |
| `autocommit` | bool | `False` | - | 是否默认自动提交 |
| `sql_mode` | str | 空串 | - | SQL 模式，如 `STRICT_TRANS_TABLES`；空值时驱动不额外设置 |

### 8.3 pg（PGSettings）

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `connection_string` | str | 空串 | - | 完整连接串；非空时**优先**于下方离散字段 |
| `host` | str | `localhost` | - | 主机 |
| `port` | int | `5432` | `gt=0` | 端口 |
| `user` | str | `postgres` | - | 用户名 |
| `password` | str | 空串 | - | 密码 |
| `database` | str | 空串 | - | 库名 |
| `sslmode` | str | `prefer` | - | SSL 模式：`disable`、`allow`、`prefer`、`require`、`verify-ca`、`verify-full` |
| `connect_timeout` | int | `5` | - | 建连最大等待（秒） |
| `application_name` | str | 空串 | - | 客户端应用名，便于数据库端监控查询来源 |
| `options` | str | 空串 | - | 会话启动参数，如 `-c timezone=UTC` |
| `target_session_attrs` | str | `any` | - | 集群路由目标：`any` 任意节点，`read-write` 仅可写主库 |

### 8.4 pool_config（DBPoolSettings）

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `print_sql` | bool | `False` | - | 是否打印 SQL，调试用 |
| `pool_size` | int | `5` | `gt=0` | 常驻连接数 |
| `max_overflow` | int | `10` | `gt=0` | 峰值额外连接数 |
| `pool_timeout` | int | `5` | `gt=0` | 取连接等待超时（秒） |
| `pool_recycle` | int | `1800` | `gt=0` | 连接回收周期（秒），规避服务端空闲断连 |
| `pool_pre_ping` | bool | `False` | - | 取连接前是否探活 |

`db.pool_config` 与 `vector.pool_config` 共用该模型，环境变量分别为 `LOONG_MEMORIES__DB__POOL_CONFIG__POOL_SIZE`、`LOONG_MEMORIES__VECTOR__POOL_CONFIG__POOL_SIZE`。

## 9. vector 分区（VectorSettings）

源文件 `config_vector.py`。承载记忆正文与向量的读写，按后端名在 `milvus` 与 `pg` 两个子分区中取连接参数。

| 字段 | 类型 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `enable` | bool | `False` | - | 是否启用向量库；为 False 时门面不初始化 |
| `backend` | Literal | `milvus` | 见下 | 后端名，对应向量库注册表键 |
| `collection` | str | 空串 | 运行时必填 | 集合名（PG 侧为表名）。留空时门面抛 `ConfigError` |
| `vector_size` | int | `1024` | `gt=0` | 向量维度，必须与 `embedding.dim` 一致 |
| `top_k` | int | `5` | `gt=0` | 检索默认返回条数 |
| `metric_type` | Literal | `COSINE` | 三选一 | 距离度量：`COSINE`、`L2`、`IP` |
| `milvus` | MilvusSettings | 默认实例 | - | Milvus 连接参数 |
| `pg` | PGVectorSettings | 默认实例 | - | PG 向量库连接参数 |
| `pool_config` | DBPoolSettings | 默认实例 | - | 连接池参数，与 `db.pool_config` 同模型 |

`backend` 可选值：`milvus`、`pg`、`milvus_async`、`pg_async`。同步入口配前两个，`AsyncMemories` 配后两个。

> 源码提示：`Literal` 声明里 `milvus` 重复出现了一次（`[... "milvus", "pg", "milvus", "milvus_async", "pg_async"]`），不影响校验结果，但枚举定义失真，建议后续去重。

### 9.1 milvus（MilvusSettings）

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `uri` | str | 空串 | 服务地址，留空用实现默认（本机 19530 端口） |
| `token` | str | 空串 | 鉴权令牌，留空表示不鉴权 |
| `db_name` | str | 空串 | 数据库名，留空使用 Milvus 默认库 |

### 9.2 pg（PGVectorSettings）

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `connection_string` | str | 空串 | 完整连接串；非空时优先于离散字段 |
| `host` | str | `localhost` | 主机 |
| `port` | int | `5432` | 端口，`gt=0` |
| `user` | str | `postgres` | 用户名 |
| `password` | str | 空串 | 密码 |
| `database` | str | 空串 | 库名 |
| `use_hnsw` | bool | `True` | 是否为向量列建立 HNSW 索引 |
| `use_diskann` | bool | `False` | 是否尝试 DiskANN 索引，缺 vectorscale 扩展时自动回退 HNSW |
| `sslmode` | str | `prefer` | SSL 模式，取值同 `db.pg.sslmode` |
| `connect_timeout` | int | `5` | 建连最大等待（秒） |
| `application_name` | str | 空串 | 客户端应用名 |
| `options` | str | 空串 | 会话启动参数 |
| `target_session_attrs` | str | `any` | 集群路由目标 |

`sslmode`、`application_name`、`options` 留空时由存储层回落到驱动默认，不会把空串直接写进连接串。

## 10. 完整示例

### 10.1 YAML（推荐）

```yaml
enable: true

embedding:
  enable: true
  provider: openai
  model: text-embedding-3-small
  dim: 1024
  batch_size: 64
  api_key: ""
  base_url: ""
  timeout: 60

llm:
  enable: true
  provider: deepseek
  model: deepseek-chat
  temperature: 0.3
  max_tokens: 0
  timeout: 60

rerank:
  enable: false
  provider: qwen
  top_k: 5
  max_documents: 100

db:
  enable: true
  backend: sqlite
  table_prefix: lm_
  sqlite:
    db_path: data/memory.db
    timeout: 30
  pool_config:
    pool_size: 5
    pool_recycle: 1800

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

### 10.2 JSON 字符串

```python
import json
from loong_memories.core.loader_config import load_config

payload = {
    "enable": True,
    "embedding": {"enable": True, "provider": "openai", "dim": 1024},
    "llm": {"enable": True, "provider": "deepseek", "model": "deepseek-chat"},
    "db": {"enable": True, "backend": "sqlite", "sqlite": {"db_path": "data/memory.db"}},
    "vector": {"enable": True, "backend": "milvus", "collection": "lm_memories", "vector_size": 1024},
}

settings = load_config(json_text=json.dumps(payload), use_env=False)
```

JSON 与 YAML 可同时传入，同名字段按 JSON 覆盖 YAML，环境变量再覆盖 JSON。

### 10.3 纯环境变量

```bash
LOONG_MEMORIES__ENABLE=true
LOONG_MEMORIES__EMBEDDING__ENABLE=true
LOONG_MEMORIES__EMBEDDING__PROVIDER=qwen
LOONG_MEMORIES__EMBEDDING__DIM=1024
LOONG_MEMORIES__LLM__ENABLE=true
LOONG_MEMORIES__LLM__PROVIDER=qwen
LOONG_MEMORIES__DB__ENABLE=true
LOONG_MEMORIES__DB__BACKEND=sqlite
LOONG_MEMORIES__DB__SQLITE__DB_PATH=data/memory.db
LOONG_MEMORIES__VECTOR__ENABLE=true
LOONG_MEMORIES__VECTOR__BACKEND=pg
LOONG_MEMORIES__VECTOR__COLLECTION=lm_memories
LOONG_MEMORIES__VECTOR__PG__DATABASE=loong
LOONG_MEMORIES__NLP__ENABLE=true
```

## 11. 校验规则与常见错误

| 报错来源 | 触发条件 | 典型报错 | 修正 |
| --- | --- | --- | --- |
| `extra=forbid` | 出现未知字段名 | `Extra inputs are not permitted` | 核对拼写，如 `db.sqlite.path` 应为 `db.sqlite.db_path` |
| 维度一致性 | `embedding.dim` 与 `vector.vector_size` 不一致且两分区都启用 | 维度不一致：embedding.dim=...，vector.vector_size=... | 改成同一数值 |
| 取值范围 | `timeout` 为 0 或负、`temperature` 超出 0 到 2、`*_size` 非正 | `Input should be greater than 0` | 按约束表调整 |
| Literal 校验 | `db.backend`、`vector.backend`、`metric_type` 取值不在枚举内 | `Input should be ...` | 从枚举表中选择 |
| YAML 缺失 | `yaml_path` 指向不存在的文件 | `YAML 配置文件不存在` | 检查路径与工作目录 |
| YAML 语法 | YAML 缩进或结构错误 | `YAML 语法错误` | 用 YAML 校验器检查 |
| JSON 顶层类型 | `json_text` 顶层是数组或标量 | 顶层必须是对象 | 改为对象 |
| 装配期缺分区 | 顶层 `enable=True` 但四必需分区的 `enable` 为 False | `ConfigError`（记忆服务未启用或分区未启用） | 补齐 enable |
| 装配期缺必填 | `db.sqlite.db_path` 或 `vector.collection` 为空 | `ConfigError` | 配置具体资源名 |
| 实现族不匹配 | 同步入口配异步后端，或反之 | `ConfigError`（需要使用同步或异步实现类） | 按入口选择有无 `_async` 后缀 |

## 12. 与门面装配的关系

`Memories.load` 与 `AsyncMemories.load` 装配时做三轮硬校验，任一不过直接抛 `ConfigError`，不会拖到运行期：

1. 总开关 `enable` 与 `REQUIRED_PARTITIONS`（embedding、llm、db、vector）必须为 True；`rerank` 与 `nlp` 可选；
2. `db.sqlite.db_path`、`vector.collection` 这类必填项不能为空；
3. 关系库、向量引擎、底层向量库三层的实现族必须与入口一致，判定依据是 `isinstance` 与协程方法，不看 `backend` 名称。

连接参数由门面按后端名组装（`_db_kwargs`、`_vector_kwargs`），底层工厂 `build_database`、`build_vector_engine`、`_backend_kwargs` 已移除，不要再按旧文档手工构造后端。

## 13. 扩展与修改指引

| 需求 | 落点 |
| --- | --- |
| 新增配置字段 | 在对应 `config_*.py` 的模型中加 `Field`，同步补默认值、约束与 `description` |
| 新增配置分区 | 在 `config_memory.py` 的 `MemorySettings` 加子模型字段，再到门面 `_assemble` 装配 |
| 新增关系库后端 | 改 `config_db.py` 的 `DBSettings.backend` Literal，并在 `db_engine` 侧提供 `BaseDB` 子类 |
| 新增向量库后端 | 改 `config_vector.py` 的 `VectorSettings.backend` Literal，并在 `vector_engine.build_vector_store` 补 if 分支 |
| 新增模型厂商 | 继承 `BaseEmbeddingProvider`、`BaseLLMProvider` 或 `BaseRerankProvider` 并声明 `backend`，配置里改 `provider` 即可切换 |
| 调整默认值 | 只改 `Field(default=...)`，同步更新本文与 README |

配置相关改动完成后，建议复核三处：本文档、`README.md` 的配置章节、`.vcl/struct.md` 的配置小节，避免文档与模型漂移。

