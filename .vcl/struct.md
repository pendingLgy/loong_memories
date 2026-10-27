# loong_memories 项目结构

> 依据：本地源码静态扫描（目录遍历 + 行数统计 + 模块头注释），不含运行时推断。
> 工作区：F:\workspace\code\python\loong_memories
> 环境：Windows (nt)、CPython 3.13.13、控制台代码页 936
> 规模：Python 文件 50 个，合计约 5674 行；tests 目录当前为空

## 1. 项目概览

| 项目 | 值 |
| --- | --- |
| 分发名 | loong_memories |
| 版本 | 0.1.0（由 `src\loong_memories\__about__.py` 动态读取） |
| 描述 | llm long memory |
| 构建后端 | hatchling |
| 布局 | src-layout |
| 声明 Python | >=3.8（运行环境 3.13.13） |
| 运行时依赖 | openai==3.26.0、PyYAML>=6.0.3、pydantic>=2.5、pydantic-settings>=2.1 |
| 可选依赖 | parsers、milvus、sqlalchemy、pg、mysql、dev |
| 测试配置 | pytest：testpaths=tests，asyncio_mode=auto，addopts 含 -m integration |
| 质量工具 | ruff（check、format）、mypy、coverage |

### 可选依赖明细

| extra | 内容 |
| --- | --- |
| parsers | unstructured>=0.12、spacy>=3.8.16、zh_core_web_sm、zh_core_web_md |
| milvus | pymilvus>=2.3 |
| sqlalchemy | SQLAlchemy>=2.0、greenlet>=3.5.6、aiosqlite>=0.19 |
| pg | psycopg[binary]>=3.1、pgvector>=0.5.1、asyncpg>=0.32.0 |
| mysql | pymysql>=1.1、aiomysql>=0.3.2 |
| dev | pytest、pytest-asyncio、pytest-cov、mypy、ruff |

## 2. 目录树（括号内为文件行数）

```text
loong_memories
├── pyproject.toml                     构建、依赖、pytest 与覆盖率配置
├── README.md
├── .idea                              IDE 工程文件（不参与分发）
├── .ruff_cache                        ruff 缓存（不参与分发）
├── src
│   └── loong_memories
│       ├── __init__.py                0    包入口（当前为空）
│       ├── __about__.py               3    版本号，hatch 动态版本来源
│       ├── loong_memories.py          0    顶层门面（当前为空，待实现）
│       ├── registry.py                51   通用类型注册表：名字映射到点分路径或类对象
│       ├── commons
│       │   └── __init__.py            0    公共工具（预留，暂无实现）
│       ├── core
│       │   ├── __init__.py            0
│       │   ├── config_db.py           75   SQLite 与数据库连接、连接池配置模型
│       │   ├── config_llm.py          34   LLM 与 Embedding 配置模型（EmbeddingSettings 在此）
│       │   ├── config_memory.py       24   总配置聚合入口，读取环境变量与配置文件
│       │   ├── config_vector.py       58   向量库配置：后端类型、集合名、连接池
│       │   └── exceptions.py          29   领域异常基类，统一上层捕获入口
│       ├── embedding
│       │   ├── __init__.py            27   包导出：BaseEmbedder、EmbedderHandler 与内置实现
│       │   ├── base.py                71   嵌入模型抽象接口，隔离具体实现
│       │   ├── embedder_handler.py    77   向量化处理器：按配置选实现，调用时实例化
│       │   ├── openai_embedder.py     78   OpenAI 嵌入实现（兼容服务复用同一请求格式）
│       │   ├── qwen_embedder.py       19   通义千问 DashScope 兼容模式实现
│       │   └── registry_embedder.py   45   嵌入模型注册表：只注册与解析，不实例化
│       ├── llm
│       │   ├── __init__.py            0
│       │   ├── base.py                71   大模型抽象接口，隔离具体实现
│       │   ├── llm_handler.py         56   大模型处理器：按配置选 provider，调用时实例化
│       │   ├── openai_llm.py          73   OpenAI 实现（qwen、deepseek 复用请求格式）
│       │   ├── deepseek_llm.py        17   DeepSeek 实现
│       │   ├── qwen_llm.py            17   通义千问 DashScope 兼容模式实现
│       │   └── registry_llm.py        46   大模型注册表：只注册与解析，不实例化
│       ├── memory
│       │   ├── __init__.py            0
│       │   ├── base.py                48   记忆层抽象契约：add、search、update、delete 四条同步流程
│       │   ├── handler.py             445  记忆层处理器：流程编排与结果组装
│       │   ├── models.py              61   数据对象：对外只暴露 pydantic 模型
│       │   ├── prompts.py             599  提示词：与 mem0 逐字一致，全项目最大文件
│       │   ├── scope.py               127  作用域构造与校验：纯函数，无外部调用
│       │   └── stores.py              278  存储薄封装：id 转换、payload 布局、批量回退
│       ├── nlp
│       │   ├── __init__.py            0
│       │   ├── lemmatization.py       148  BM25 关键词词形还原（spaCy lemmatizer）
│       │   └── spacy.py               100  spaCy 模型加载与线程安全封装
│       ├── storage
│       │   ├── memories
│       │   │   ├── __init__.py            0
│       │   │   ├── base.py                109  向量库抽象接口（score 越大越相似）
│       │   │   ├── milvus_store.py        460  Milvus 实现：集合、稠密与稀疏向量、AUTOINDEX
│       │   │   ├── pg_store.py            690  PostgreSQL 与 pgvector 实现
│       │   │   ├── registry_vector.py     46   向量库后端注册表
│       │   │   └── vector_engine.py       286  向量引擎：按后端名构造并统一读写
│       │   └── message
│       │       ├── __init__.py            0
│       │       ├── base.py                54   关系库抽象接口：历史记录与消息持久化
│       │       ├── db_engine.py           383  通用实现：基于 SQLAlchemy Core
│       │       ├── mysql_db.py            298  MySQL 实现（PyMySQL 优先）
│       │       ├── pg_db.py               262  PostgreSQL 实现（psycopg3 优先）
│       │       ├── registry_db.py         45   数据库后端注册表
│       │       ├── schema.py              86   表结构定义（SQLAlchemy Core）
│       │       └── sqlite_db.py           202  SQLite 实现（sqlite3 与 aiosqlite）
│       └── util
│           ├── __init__.py            0
│           └── snowflake.py           76   雪花 ID 生成器（64 位、趋势递增）
└── tests                              空目录，无测试文件
```

## 3. 分层与职责

| 层次 | 位置 | 职责 |
| --- | --- | --- |
| 门面 | loong_memories.py（空） | 对外统一入口，当前未实现 |
| 配置 | core | pydantic-settings 配置模型，聚合 DB、LLM、Embedding、Vector |
| 业务编排 | memory | add、search、update、delete 四条同步流程 |
| 能力适配 | llm、embedding | 大模型与向量化的抽象、注册表、处理器、厂商实现 |
| 持久化 | storage.message、storage.memories | 关系库存消息，向量库存记忆 |
| 支撑 | nlp、util、commons | 分词与词形还原、雪花 ID、公共预留 |

依赖方向：memory 依赖 llm、embedding、storage 与 core；storage 只依赖抽象基类，后端实现按需惰性导入，避免环导入。

## 4. 关键设计模式

### 4.1 注册表 + 处理器（llm、embedding）

- `registry_*.py` 只做名字到实现（类对象或点分路径）的登记与解析，不做实例化，模块内不持有实例。
- `*_handler.py` 按配置选择 provider，在调用时才实例化；支持注入模型句柄以便测试。
- 注册两种方式：继承基类并声明 backend 自动注册，或调用 register_llm、register_embedder 显式登记。

### 4.2 引擎 + 后端（storage）

- `storage.message`：`db_engine.build_database` 按后端名构造，通用逻辑基于 SQLAlchemy Core，方言差异由 `sqlite_db`、`pg_db`、`mysql_db` 适配，表结构统一定义在 `schema.py`。
- `storage.memories`：`vector_engine` 按后端名构造向量库，契约在 `base.py`（score 越大越相似，距离度量需先转相似度），实现为 `milvus_store`、`pg_store`。
- 缺驱动时仅在实例化时报错，模块本身可正常导入。

### 4.3 记忆层内部结构

- `base.py`：四条流程的抽象契约，与 storage 子包同构。
- `handler.py`：只做流程编排与结果组装，不碰存储细节。
- `stores.py`：统一 id 转换、payload 布局与批量回退策略，只依赖 DBEngine 与 VectorEngine 的抽象方法。
- `scope.py`：作用域构造与校验，全部为纯函数；`user_id`、`agent_id`、`run_id` 是唯一身份来源，metadata 中同名字段一律剥离。
- `prompts.py`：与 mem0 保持逐字一致，不改写。
- `models.py`：对外只暴露 pydantic 模型，避免裸字典跨层传递。

## 5. 待补充与注意事项

- `tests` 目录为空：pytest 已配置 testpaths、asyncio_mode 与 integration 标记，但无用例。
- `loong_memories.py`、包级 `__init__.py`、`commons` 均为空文件，尚无对外导出与门面实现。
- `config_llm.py` 内含 `EmbeddingSettings`，与目录命名语义略有偏差（embedding 配置放在 llm 配置文件）。
- 声明支持 Python 3.8，实际运行环境为 3.13；`pydantic>=2.5` 与 `openai==3.26.0` 是比较强的版本约束。
- 本文档结论均来自静态阅读与文件扫描，未执行任何测试用例或运行时校验。

## 6. 扩展落点建议

- 若要实现「对话历史滚动压缩与长文本记忆对齐」：入口落在 `memory.handler`，提示词落在 `memory.prompts`，阈值等配置追加到 `core.config_memory`，摘要消息的识别与过滤放在 `memory.scope` 或独立纯函数模块。
- 若要新增存储后端：实现 `storage.message.base` 或 `storage.memories.base` 的子类并声明 backend，无需改动引擎与注册表。
- 若要新增模型厂商：OpenAI 兼容服务直接继承 `openai_llm` 或 `openai_embedder`，只覆盖默认模型、接口地址、Key 环境变量与批量上限。

