# loong_memories 项目结构

> 依据：本地源码静态扫描（目录遍历 + 行数统计 + 模块头注释与公共签名阅读），不含运行时推断。
> 工作区：`F:\workspace\code\python\loong_memories`
> 环境：Windows (nt)、CPython 3.13.13、cmd.exe、代码页 936

## 1. 规模

| 范围 | 文件数 | 行数 |
| --- | --- | --- |
| `src/loong_memories` | 51 个 `.py` | 6129 行 |
| `tests` | 9 个 `.py` | 3630 行 |
| 仓库根 | `pyproject.toml`、`.gitignore`、`README.md`、`DEVELOPMENT.md` | 文档另行维护 |

其中 10 个文件为 0 行占位（9 个包级 `__init__.py` 与 `loong_memories.py` 门面）。最大五个文件：`pg_store.py`(730)、`prompts.py`(599)、`milvus_store.py`(464)、`handler.py`(445)、`db_engine.py`(414)。

## 2. 项目概览

| 项目 | 值 |
| --- | --- |
| 分发名 | loong_memories（描述：llm long memory） |
| 版本 | 0.1.0，由 `src/loong_memories/__about__.py` 动态读取 |
| 构建后端 | hatchling，src-layout |
| 声明 Python | `>=3.8`（实际运行 3.13.13） |
| 运行时依赖 | openai、PyYAML、pydantic、pydantic-settings |
| 可选依赖 | parsers、milvus、sqlalchemy、pg、mysql、dev |
| 测试配置 | testpaths=tests、asyncio_mode=auto、addopts=`-s -q -rs -m 'integration'` |
| 质量工具 | ruff（check、format）、mypy、coverage（branch 与 parallel） |

### 可选依赖明细

| extra | 内容 |
| --- | --- |
| parsers | unstructured、spacy、zh_core_web_sm、zh_core_web_md |
| milvus | pymilvus |
| sqlalchemy | SQLAlchemy、greenlet、aiosqlite |
| pg | psycopg[binary]、pgvector、asyncpg |
| mysql | pymysql、aiomysql |
| dev | pytest、pytest-asyncio、pytest-cov、mypy、ruff |

## 3. 目录树（括号内为行数）

```text
loong_memories
├── pyproject.toml                        构建、依赖、pytest、覆盖率配置
├── README.md                             对外说明（安装、快速开始、配置、后端矩阵）
├── DEVELOPMENT.md                        开发手册（环境、分层、测试、已知问题）
├── .vcl/struct.md                         本文件：结构快照
├── src
│   └── loong_memories
│       ├── __init__.py                   0      包入口（空）
│       ├── __about__.py                  3      版本号，hatch 动态版本来源
│       ├── loong_memories.py             0      顶层门面（空，待实现）
│       ├── registry.py                  51      通用类型注册表：名字到点分路径或类对象
│       ├── commons/__init__.py             0      公共工具（预留）
│       ├── core
│       │   ├── __init__.py               0
│       │   ├── config_db.py            125      SQLite / MySQL / PG 连接与连接池配置
│       │   ├── config_llm.py            47      LLMSettings 与 EmbeddingSettings
│       │   ├── config_memory.py         24      MemorySettings 总配置，四分区聚合
│       │   ├── config_vector.py         80      向量库配置：milvus / pg 与集合、度量
│       │   ├── exceptions.py            29      领域异常基类与七个子类
│       │   └── loader_config.py        181      YAML / JSON / 环境变量三级叠加装载
│       ├── embedding
│       │   ├── __init__.py              27      包导出
│       │   ├── base.py                  71      BaseEmbedder 抽象接口
│       │   ├── embedder_handler.py      77      分批编码、维度校验、模型句柄可注入
│       │   ├── openai_embedder.py       78      OpenAI 嵌入实现
│       │   ├── qwen_embedder.py         19      DashScope 兼容模式实现
│       │   └── registry_embedder.py     44      注册表：只登记与解析
│       ├── llm
│       │   ├── __init__.py               0
│       │   ├── base.py                  71      BaseLLMProvider 抽象接口
│       │   ├── llm_handler.py           56      provider 装配与 chat / complete
│       │   ├── openai_llm.py            73      OpenAI 实现（qwen、deepseek 复用请求格式）
│       │   ├── deepseek_llm.py          17      DeepSeek 实现
│       │   ├── qwen_llm.py              17      通义千问实现
│       │   └── registry_llm.py          48      注册表：只登记与解析
│       ├── memory
│       │   ├── __init__.py               0
│       │   ├── base.py                  48      MemoryBase：四流程抽象契约
│       │   ├── handler.py              445      MemoryHandler：流程编排与结果组装
│       │   ├── models.py                61      MemoryItem / Add / Search / Update / Delete 结果
│       │   ├── prompts.py              599      提示词，与 mem0 逐字一致
│       │   ├── scope.py                127      作用域构造与校验（纯函数）
│       │   └── stores.py               283      MemoryStore：id 转换、payload 布局、批量回退
│       ├── nlp
│       │   ├── __init__.py               0
│       │   ├── lemmatization.py        148      BM25 关键词词形还原
│       │   └── spacy.py                100      spaCy 模型加载与线程安全封装
│       ├── storage
│       │   ├── memories
│       │   │   ├── __init__.py            0
│       │   │   ├── base.py              109      VectorStoreBase 契约（score 越大越相似）
│       │   │   ├── milvus_store.py      464      Milvus 实现，含 BM25 稀疏向量与异步客户端
│       │   │   ├── pg_store.py          730      pgvector 实现，含索引与过滤构造
│       │   │   ├── registry_vector.py    45      注册表
│       │   │   └── vector_engine.py     276      VectorEngine：构造后端并统一读写
│       │   └── message
│       │       ├── __init__.py            0
│       │       ├── base.py               50      BaseDB 契约：历史与消息
│       │       ├── db_engine.py         414      通用实现（DBEngine / AsyncDBEngine / build_database）
│       │       ├── mysql_db.py          341      MySQL 同步与异步
│       │       ├── pg_db.py             316      PostgreSQL 同步与异步
│       │       ├── registry_db.py        42      注册表
│       │       ├── schema.py             86      表结构与 HistoryRecord / Message 数据对象
│       │       └── sqlite_db.py         231      SQLite 同步与异步
│       └── util
│           ├── __init__.py               0
│           └── snowflake.py             76      64 位雪花 ID
└── tests
    ├── conftest.py                      212      命令行开关、标记注册、四分区配置样例
    ├── test_loader_config.py            513      配置装载：默认值 / YAML / JSON / ENV
    ├── test_embedding.py                281      嵌入契约、注册表、handler
    ├── test_llm.py                      299      大模型契约、注册表、handler
    ├── test_sqlite_db.py                422      关系库契约与业务矩阵（默认执行）
    ├── test_mysql_db.py                 513      同上（integration 门控）
    ├── test_pg_db.py                    503      同上（integration 门控）
    ├── test_milvus_store.py             408      向量库契约与业务矩阵（integration）
    └── test_pg_store.py                 479      同上（integration）
```

## 4. 分层与职责

| 层次 | 位置 | 职责 | 对外契约 |
| --- | --- | --- | --- |
| 门面 | `loong_memories.py`（空） | 对外统一入口 | 未实现 |
| 配置 | `core` | pydantic 配置模型、多来源装载、领域异常 | `load_config` → `MemorySettings` |
| 业务编排 | `memory` | add / search / update / delete 四流程 | `MemoryBase` → `MemoryHandler` |
| 能力适配 | `llm`、`embedding` | 抽象、注册表、处理器、厂商实现 | `BaseLLMProvider` / `BaseEmbedder` |
| 持久化 | `storage.message`、`storage.memories` | 关系库存消息与历史，向量库存记忆 | `BaseDB` / `VectorStoreBase` |
| 支撑 | `nlp`、`util`、`commons` | 词形还原、雪花 ID、公共预留 | 纯函数 |

依赖方向：`memory` → `llm`、`embedding`、`storage`、`core`；`llm`、`embedding`、`storage` → `core`（`storage` 另用 `util`、`nlp`）；`core` 只依赖第三方库。方言实现全部在函数内惰性导入，避免导入环与可选依赖硬绑定。

## 5. 关键设计模式

### 5.1 注册表加工厂（llm、embedding、storage）

- `registry_*.py` 只做名字到实现（类对象或点分路径）的登记与解析，模块内不持有实例。
- 登记两条路径：继承基类并声明 `backend`（`__init_subclass__` 自动登记），或调用 `register_*` / `register_*_provider`。
- 实例化点在 `*_handler.py`（大模型、嵌入）、`vector_engine.build_vector_store`、`db_engine.build_database`。
- 名字解析失败抛 `ConfigError`，并列出全部可用名字；可选依赖缺失只在实例化时报 `OptionalDependencyError`。

### 5.2 引擎 + 方言（storage）

- `storage.message`：`DBEngine` 与 `AsyncDBEngine` 承载通用逻辑（SQLAlchemy Core），方言子类只提供连接串与建表检查；表结构统一定义在 `schema.py`。
- `storage.memories`：`VectorStoreBase` 定义九个方法契约，`VectorEngine` 统一入参校验与日志，`milvus_store` / `pg_store` 各提供同步与异步实现。
- 契约要点：`search` 返回 score 越大越相似；`keyword_search` 不支持时返回 `None` 而非抛异常；`list` 按既有约定包裹返回。

### 5.3 记忆层内部分工

- `handler.py`：只做编排与结果组装，不碰存储细节；构造期强制校验四个依赖。
- `stores.py`：统一 id 转换、payload 布局（核心键与业务元数据分离）、批量失败回退逐条、历史写入只告警不阻断。
- `scope.py`：作用域与查询校验全部为纯函数，`user_id` / `agent_id` / `run_id` 是唯一身份来源，metadata 同名字段一律剥离。
- `prompts.py`：与 mem0 逐字一致，不改写。
- `models.py`：对外只暴露 pydantic 模型（`MemoryItem` 与四个结果包装），禁止裸字典跨层。

### 5.4 数据流

```text
add      messages → 归一化 → 召回既有记忆（向量 + 关键词）→ 大模型抽事实
         → 指纹去重 → 批量嵌入 → 写向量库 → 写历史 → 写会话消息
search   query → 嵌入 → 双路召回（recall_k = max(top_k*4, 60)）
         → 融合打分（语义 + 0.5 * 关键词）→ 阈值过滤 → top_k 截断
update   get → 合并 payload → 正文变化才重算向量 → 更新向量库 → 写 UPDATE 历史
delete   get → 向量库删除 → 写 DELETE 历史
```

## 6. 待补充与注意事项

- `loong_memories.py`、包级 `__init__.py`、`commons` 仍为空文件，尚无对外门面与导出；装配流程需手工调用私有 `_start()`。
- `memory/handler.py` 的 `_persist_facts` 缺少 `next_id` 与 `Message` 的导入，落库分支会抛 `NameError`。
- `config_llm.py` 内含 `EmbeddingSettings`，与目录命名语义略有偏差（embedding 配置放在 llm 配置文件）。
- `vector` 分区的 `backend` 只放行 `milvus` 与 `pg`，异步实现已登记但配置不可达。
- 声明支持 Python 3.8，实际运行 3.13；`pydantic>=2.5` 与 `openai==3.26.0` 是较强的版本约束。
- 本文档结论来自静态阅读与文件扫描，未执行任何测试用例或运行时校验。

## 7. 扩展落点

| 需求 | 落点 |
| --- | --- |
| 新增存储后端 | 继承 `BaseDB` / `VectorStoreBase` 并声明 `backend`，再在 `config_db.py` / `config_vector.py` 的 `Literal` 放行 |
| 新增模型厂商 | 继承 `OpenAILLMProvider` / `OpenAIEmbedder`，只覆盖 `backend`、`default_model`、`base_url`、`api_key_env`、`max_batch_size` |
| 新增配置项 | `core/config_*.py` 的 pydantic 模型，字段必须写 `description`，模型保持 `extra="forbid"` |
| 对话历史滚动压缩 | 门控与裁剪进 `memory/handler.py`，摘要复用 `LLMHandler`，提示词进 `memory/prompts.py`，阈值进 `core/config_memory.py`，摘要识别与轮次切分做纯函数模块 |
| 顶层门面 | `loong_memories.py`：由 `MemorySettings` 一次性装配四组件并暴露四流程 |
