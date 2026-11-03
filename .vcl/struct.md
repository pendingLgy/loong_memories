# loong_memories 项目结构与架构分析

> 依据：本地源码静态扫描（目录遍历 + 行数统计 + AST 提取公共签名），不含运行时推断。
> 环境：Windows 11 (10.0.22631)、CPython 3.13.13、cmd.exe、代码页 936；源码 UTF-8 无 BOM、CRLF。
> 时点：git HEAD 之后的工作区状态（门面 `memories.py` 已实现、`nlp` 分区已接入、`build_database` 与 `build_vector_engine` 已移除）。

## 1. 规模与基本信息

### 1.1 源码规模

| 范围 | 文件数 | 行数 | 备注 |
| --- | ---: | ---: | --- |
| `src/loong_memories` | 61 | 9071 | 含许可头与 docstring |
| `tests` | 13 | 7998 | 静态 `def test_` 计数 619 项，未展开参数化 |
| 仓库根 | pyproject.toml、README.md、DEVELOPMENT.md、LICENSE | - | 文档另计 |

最大十个文件：`pg_store.py`(945)、`handler.py`(816)、`prompts.py`(617)、`stores.py`(612)、`milvus_store.py`(568)、`db_engine.py`(402)、`mysql_db.py`(352)、`memories.py`(344)、`pg_db.py`(342)、`vector_engine.py`(326)。

唯一位 0 行文件：`reranks/__init__.py`。仅 `embedding/__init__.py`(41) 有导出；`llm`、`memory`、`nlp`、`storage` 两个子包、`util`、`core`、`commons` 的 `__init__.py` 只有许可头。包根 `__init__.py`(13) 仍只有许可头，**刻意不导出 `memories`**，保证 `import loong_memories` 不牵入 sqlalchemy。

目录内残留 `__pycache__` 下的 `cpython-312` 字节码，与当前解释器 3.13 不一致，属历史构建产物。

### 1.2 分发与构建

| 项 | 值 |
| --- | --- |
| 分发名 | loong_memories（描述：llm long memory） |
| 版本 | 0.1.0，由 `src/loong_memories/__about__.py` 动态读取 |
| 构建后端 | hatchling，src-layout |
| 声明 Python | `>=3.8`（实际运行 3.13.13，代码已用 `dict[str, Any]` 与 `X | None` 语法） |
| 许可 | Apache-2.0 |
| 测试配置 | testpaths=tests、asyncio_mode=auto、asyncio_default_fixture_loop_scope=module、addopts=`-s -q -rs`、开启 log_cli |
| 自定义标记 | unit、integration、integration_ai（conftest 的 pytest_configure 注册）；CLI 选项 `--integration`、`--integration_ai` |

`addopts` 中的 `-m integration` 已被注释，默认收集全部用例；集成用例靠文件内的参数门控自行 `skip`。

### 1.3 依赖分组

| 分组 | 内容 |
| --- | --- |
| 运行时 | openai==3.26.0、PyYAML 不低于 6.0.3、pydantic 不低于 2.5、pydantic-settings 不低于 2.1、requests 不低于 2.31 |
| parsers | unstructured 不低于 0.12、spacy 不低于 3.8.16、zh_core_web_sm、zh_core_web_md（URL wheel） |
| milvus | pymilvus 不低于 2.3 |
| sqlalchemy | SQLAlchemy 不低于 2.0、greenlet 不低于 3.5.6、aiosqlite 不低于 0.19 |
| pg | psycopg[binary] 不低于 3.1、pgvector 不低于 0.5.1、asyncpg 不低于 0.32.0 |
| mysql | pymysql 不低于 1.1、aiomysql 不低于 0.3.2 |
| dev | pytest 不低于 7.4、pytest-asyncio 不低于 1.4.0、pytest-cov 不低于 4.1、mypy 不低于 1.8、ruff 不低于 0.1.9 |

hatch 环境：`default` = parsers + sqlalchemy + dev；`loong_memories_dev` = parsers + sqlalchemy + pg + mysql + milvus + dev；`types` 只装 mypy。spaCy 模型走直接 URL，因此 `allow-direct-references = true`。

## 2. 完整目录树解析

按文件与目录逐条展开，括号内为该文件行数。

```text
loong_memories/
├─ pyproject.toml                            构建后端、依赖分组、hatch 环境、pytest 与 coverage
├─ README.md                                 对外使用说明
├─ DEVELOPMENT.md                            开发手册
├─ LICENSE                                   Apache-2.0 全文
├─ .env                                      本机环境变量（不入库）
├─ .gitignore                                忽略规则
├─ test.db                                   SQLite 用例遗留库文件（未跟踪）
├─ .vcl/
│  ├─ struct.md                              本文件：项目结构与架构分析
│  ├─ defect_report.md                       缺陷清单、根因与修复补丁
│  └─ skills/                               技能索引与规范文档
├─ src/
│  └─ loong_memories/
│     ├─ __init__.py(13)                     仅许可头，无导出；刻意不引入 memories
│     ├─ __about__.py(16)                    版本号 0.1.0，供 hatch 动态读取
│     ├─ memories.py(344)                    **统一装配门面：Memories、AsyncMemories、11 个模块级私有函数**
│     ├─ registry.py(61)                     通用 Registry 与 Ref 类型别名
│     ├─ commons/
│     │  └─ __init__.py(13)                  空包，仅许可头
│     ├─ core/
│     │  ├─ __init__.py(13)                  仅许可头
│     │  ├─ config_db.py(140)                DBSettings、SQLiteSettings、MySQLSettings、PGSettings、DBPoolSettings
│     │  ├─ config_llm.py(106)               EmbeddingSettings、LLMSettings、RerankSettings
│     │  ├─ config_memory.py(80)             MemorySettings 总配置（六分区），校验维度一致
│     │  ├─ config_nlp.py(27)                **NlpSettings：enable 与 nlp_name**
│     │  ├─ config_vector.py(95)             VectorSettings、MilvusSettings、PGVectorSettings
│     │  ├─ exceptions.py(50)                RagDataError 及 ConfigError、DataError、StorageError 等子类
│     │  └─ loader_config.py(201)            YAML、JSON、环境变量三级深合并，交给 MemorySettings 校验
│     ├─ embedding/
│     │  ├─ __init__.py(41)                  包级导出：抽象基类、处理器、内置实现与注册函数
│     │  ├─ base.py(88)                      BaseEmbedder 抽象，声明 backend、default_model、max_batch_size
│     │  ├─ embedder_handler.py(94)          EmbedderHandler：分批编码并按 dim 逐条校验
│     │  ├─ openai_embedder.py(96)           OpenAI 向量化实现
│     │  ├─ qwen_embedder.py(37)             DashScope 向量化实现，max_batch_size 为 10
│     │  └─ registry_embedder.py(61)         嵌入实现注册表与解析函数
│     ├─ llm/
│     │  ├─ __init__.py(13)                  仅许可头
│     │  ├─ base.py(88)                      BaseLLMProvider 抽象
│     │  ├─ deepseek_llm.py(34)              DeepSeek 实现，兼容 OpenAI 协议
│     │  ├─ llm_handler.py(73)               LLMHandler：chat 与 complete
│     │  ├─ openai_llm.py(91)                OpenAI 对话实现
│     │  ├─ qwen_llm.py(35)                  Qwen 实现，兼容 OpenAI 协议
│     │  └─ registry_llm.py(65)              LLM 实现注册表与解析函数
│     ├─ memory/
│     │  ├─ __init__.py(13)                  仅许可头
│     │  ├─ base.py(96)                      MemoryBase 抽象：get、page_query、add、search、update、delete
│     │  ├─ handler.py(816)                  流程编排：MemoryHandler 与 AsyncMemoryHandler，接收 nlp_setting
│     │  ├─ models.py(96)                    MemoryItem 与六种结果对象（时间字段 _tz 后缀）
│     │  ├─ prompts.py(617)                  抽取提示词常量与渲染函数
│     │  ├─ scope.py(135)                    作用域构造、会话键、参数校验，全为纯函数
│     │  └─ stores.py(612)                   MemoryStore 与 AsyncMemoryStore，双路召回与打分
│     ├─ nlp/
│     │  ├─ __init__.py(13)                  仅许可头
│     │  ├─ lemmatization.py(164)            中英文词形还原，供 BM25 使用
│     │  └─ spacy.py(146)                    spaCy 管线双缓存、失败哨兵与句子边界
│     ├─ reranks/
│     │  ├─ __init__.py(0)                   空文件，无任何内容
│     │  ├─ base.py(98)                      BaseRerankProvider 抽象
│     │  ├─ qwen_rerank.py(153)              DashScope 原生 text-rerank 调用与失败降级
│     │  ├─ registry_rerank.py(61)           重排实现注册表与解析函数
│     │  └─ rerank_handler.py(64)            RerankHandler：按配置装配并透传调用
│     ├─ storage/
│     │  ├─ memories/
│     │  │  ├─ __init__.py(13)               仅许可头
│     │  │  ├─ base.py(188)                  VectorStoreBase 八个抽象方法、VectorStoreResult 与结果转换
│     │  │  ├─ milvus_store.py(568)          MilvusVectorStore 与 AsyncMilvusVectorStore
│     │  │  ├─ pg_store.py(945)              PGVectorStore 与 AsyncPGVectorStore，表 vut_vectors
│     │  │  ├─ registry_vector.py(63)        向量库注册表，门面经此解析实现类
│     │  │  └─ vector_engine.py(326)         VectorEngine 与 AsyncVectorEngine，构造后端并校验入参
│     │  └─ message/
│     │     ├─ __init__.py(13)               仅许可头
│     │     ├─ base.py(67)                   BaseDB 抽象，声明 backend 后自动登记
│     │     ├─ db_engine.py(402)             DBEngine 与 AsyncDBEngine，通用 CRUD 与建表
│     │     ├─ mysql_db.py(352)              MySQLDB 与 AsyncMySQLDB，驱动双回退
│     │     ├─ pg_db.py(342)                 PGDB 与 AsyncPGDB，驱动在 psycopg3 与 psycopg2 间回退
│     │     ├─ registry_db.py(57)            关系库注册表，门面经 resolve_db 解析
│     │     ├─ schema.py(120)                history 与 messages 两表定义，tenant_id 均 NOT NULL
│     │     └─ sqlite_db.py(249)             SQLiteDB 与 AsyncSQLiteDB
│     └─ util/
│        ├─ __init__.py(13)                  仅许可头
│        ├─ datetime_util.py(30)             ensure_datetime、now_datetime、now_iso、datetime_to_isoformat
│        ├─ numberic_util.py(23)             str_to_int 与 str_to_float，空值返回 None，解析失败抛 ParserError
│        ├─ scoring.py(141)                  BM25 归一、sigmoid 参数选择与融合打分
│        ├─ snowflake.py(94)                 雪花 ID 生成器与模块级 next_id
│        └─ str_util.py(6)                   memory_hash（MD5 指纹）
└─ tests/
   ├─ conftest.py(245)                      路径注入、两个 CLI 选项、三个标记、六份配置字典与 Settings 对象
   ├─ test_embedding.py(283)                注册表、内置 embedder 与 handler，35 项
   ├─ test_llm.py(299)                      注册表、内置 provider 与 handler，38 项
   ├─ test_loader_config.py(513)            配置三级来源与异常分支，84 项
   ├─ test_sqlite_db.py(521)                关系库契约与业务矩阵，50 项
   ├─ test_integration_vector_engine.py(605)  引擎契约与后端矩阵，51 项
   ├─ test_integration_pg_store.py(785)     PG 向量库契约与业务矩阵，56 项
   ├─ test_integration_milvus_store.py(713) Milvus 向量库契约与业务矩阵，58 项
   ├─ test_integration_mysql_db.py(613)     MySQL 关系库契约与业务矩阵，49 项
   ├─ test_integration_pg_db.py(603)        PG 关系库契约与业务矩阵，49 项
   ├─ test_integration_stores.py(992)       MemoryStore 读写、审计与召回，45 项
   ├─ test_integration_ai_memory_handler.py(879)  处理器端到端，36 项
   └─ test_integration_ai_memories.py(947) **门面端到端：契约 59 项 + 真实库 9 项，共 68 项**
```

### 目录规模汇总

| 目录 | 文件数 | 行数 |
| --- | ---: | ---: |
| src/loong_memories/core | 8 | 712 |
| src/loong_memories/embedding | 6 | 417 |
| src/loong_memories/llm | 7 | 399 |
| src/loong_memories/memory | 7 | 2385 |
| src/loong_memories/nlp | 3 | 323 |
| src/loong_memories/reranks | 5 | 376 |
| src/loong_memories/storage/memories | 6 | 2103 |
| src/loong_memories/storage/message | 8 | 1602 |
| src/loong_memories/util | 6 | 307 |
| src/loong_memories/commons | 1 | 13 |
| 包根（init、about、memories、registry） | 4 | 434 |
| 合计 | 61 | 9071 |

## 3. 分层与依赖方向

```text
memories（门面）  装配入口：Memories 与 AsyncMemories，只做配置校验与组件构造
   ├── core.loader_config              配置三级深合并 → MemorySettings
   ├── embedding、llm、reranks 的处理器  能力层：按配置构造 embedder、llm、reranker
   ├── storage.message.registry_db       resolve_db 解析关系库实现类
   ├── storage.memories.registry_vector  resolve_vector_store 解析向量库实现类
   └── memory.handler                    产出 MemoryHandler 与 AsyncMemoryHandler

memory/handler        流程编排：get、page_query、add、search、update、delete
   ├── memory/stores          MemoryStore 与 AsyncMemoryStore：向量库与关系库组合封装
   │      ├── storage/memories/vector_engine   VectorEngine 与 AsyncVectorEngine
   │      │      └── storage/memories 下的 milvus_store 与 pg_store
   │      └── storage/message/db_engine        DBEngine 与 AsyncDBEngine
   │             └── storage/message 下的 sqlite_db、mysql_db、pg_db
   ├── memory/scope        纯函数：作用域构造、会话键、参数校验
   ├── memory/prompts      提示词常量与渲染
   ├── embedding/embedder_handler   EmbedderHandler
   ├── llm/llm_handler              LLMHandler
   ├── reranks/rerank_handler       RerankHandler
   └── core.config_nlp              NlpSettings（词形还原开关与模型名）

core  只承载配置、异常、装载与注册表，不依赖任何领域服务。
util 与 nlp 为纯函数层，被 memory 与 storage 复用。
storage 的方言实现全部在函数内惰性导入，缺驱动不影响包导入。
```

### 3.1 解耦手段

| 手段 | 落点 | 效果 |
| --- | --- | --- |
| 自动注册表 | `VectorStoreBase.__init_subclass__`、`BaseDB.__init_subclass__` | 子类声明 backend 即登记，无需显式注册调用 |
| 门面经注册表取类 | `memories._store_class` 与 `resolve_db` | 关系库完全走注册表；向量库经 `_store_class` 归一后走注册表 |
| 惰性导入 | `vector_engine.build_vector_store`、各方言模块 | 缺 pymilvus 或 psycopg 时仍可 import 包 |
| 抽象基类 | MemoryBase、VectorStoreBase、BaseDB、BaseEmbedder、BaseLLMProvider、BaseRerankProvider | 上层只依赖契约 |
| 配置驱动 | `core.loader_config` 加 pydantic 模型 | YAML、JSON、env 三层深合并 |
| 实现族隔离 | `_db_is_async`、`_engine_is_async`、`_store_is_async` 三个判定函数 | 同步入口不可能拿到异步实现，反之亦然 |
| 公共工具 | `util.str_util`、`util.numberic_util`、`util.datetime_util` | 指纹、ID 归一与时间转换逻辑单点维护 |

已知不一致：`vector_engine.build_vector_store` 仍用 if 分支直连 `MilvusVectorStore` 与 `PGVectorStore`，不查注册表，新增向量后端必须同时改配置 `Literal` 与该分支；关系库侧已由门面直接 `resolve_db` 解析，注册表是生效的。

## 4. 核心模块职责拆解

### 4.1 配置与初始化层（core）

| 模型 | 关键字段（默认值） |
| --- | --- |
| EmbeddingSettings | enable(False)、provider(openai)、model、batch_size(128)、dim(1024)、api_key、base_url、timeout(60，必须大于 0) |
| LLMSettings | enable(False)、provider、model、temperature(0.7，0 到 2)、max_tokens(0 表示不传)、api_key、base_url、timeout(60) |
| RerankSettings | enable(False)、provider(qwen)、model、top_k(5)、max_tokens(0)、max_documents(100)、api_key、base_url、timeout(60) |
| NlpSettings | enable(False)、nlp_name(zh_core_web_md) |
| MemorySettings | extra=forbid；enable 总开关；embedding、llm、rerank、db、vector、nlp 六个子配置 default_factory；settings_customise_sources 只保留 init_settings；校验 embedding.dim 与 vector.vector_size 一致 |
| DBSettings | enable、backend（Literal：sqlite、mysql、pg 及三个 _async 变体）、sqlite、mysql、pg、table_prefix、pool_config |
| SQLiteSettings | db_path（默认空串，运行时校验）、timeout(30)、isolation_level、check_same_thread(False) |
| MySQLSettings | host、port、user、password、database、charset、connect_timeout、autocommit、sql_mode |
| PGSettings | host、port、user、password、database、connection_string、sslmode、connect_timeout、application_name、options、target_session_attrs |
| DBPoolSettings | print_sql、pool_size(5)、max_overflow(10)、pool_timeout(5)、pool_recycle(1800)、pool_pre_ping |
| VectorSettings | enable、backend（Literal：milvus、pg、milvus、milvus_async、pg_async）、collection、vector_size(1024)、top_k(5)、metric_type、milvus、pg、pool_config |
| MilvusSettings | uri、token、db_name |
| PGVectorSettings | connection_string、host、port、user、password、database、use_hnsw(True)、use_diskann(False)、sslmode 等高级参数 |

装载入口 `load_config(json_text=None, yaml_path=None, use_env=True, env_prefix=LOONG_MEMORIES__)`：YAML 到 JSON 到环境变量依次深合并，再交给 `MemorySettings.model_validate`，失败统一抛 ConfigError。辅助函数：load_config_from_json、load_config_from_yaml、load_config_from_env、_deep_merge、_ensure_mapping、_set_nested、_coerce_env_value、_import_yaml。

**`VectorSettings.backend` 的 Literal 里 `milvus` 写了两次**（`milvus`、`pg`、`milvus`、`milvus_async`、`pg_async`），属笔误，虽不影响校验但会让枚举定义失真，建议去重。

异常层次（`core/exceptions.py`，50 行）：RagDataError 派生出 ConfigError、DataError、OptionalDependencyError（再到 ParserDependencyError）、ParserError、EmbeddingError、LLMError、StorageError。

### 4.2 工具与底层支撑层（util、nlp、registry）

| 文件 | 内容 |
| --- | --- |
| `util/snowflake.py` | EPOCH_MS=1704067200000；SnowflakeGenerator(datacenter_id, worker_id) 加锁生成；模块级 next_id 用进程内单例 |
| `util/str_util.py` | memory_hash(text)：MD5 指纹，用于 add 去重与 update 变更判定 |
| `util/numberic_util.py` | str_to_int 与 str_to_float：空值与空串返回 None，解析失败抛 ParserError |
| `util/datetime_util.py` | ensure_datetime、now_datetime、now_iso、datetime_to_isoformat |
| `util/scoring.py` | get_bm25_params、normalize_bm25、ENTITY_BOOST_WEIGHT=0.5、score_and_rank |
| `nlp/spacy.py` | DEFAULT_MODEL=zh_core_web_md；双缓存共用一把 Lock，失败写哨兵 _FAILED；load_nlp、get_nlp_lemma、clear_cache、ensure_sentence_boundaries |
| `nlp/lemmatization.py` | lemmatize_for_bm25、lemmatize_zh_for_bm25，保留下划线与连字符 |
| `registry.py` | Registry(dict)：register、decorator、names、resolve（点分路径惰性导入）；Ref 为 str 或 type 的联合别名 |

### 4.3 模型能力层（embedding、llm、reranks）

三套能力同构：注册表只登记实现类，实例化在 handler 首次调用时完成，可注入 model 绕过配置。

| 能力 | 抽象基类 | 内置 backend | 配置键 | 关键上限 |
| --- | --- | --- | --- | --- |
| 向量化 | BaseEmbedder | openai、qwen | embedding.provider | qwen 的 max_batch_size=10 |
| 大模型 | BaseLLMProvider | openai、qwen、deepseek | llm.provider | max_tokens=0 表示不传 |
| 重排 | BaseRerankProvider | qwen（DashScope text-rerank） | rerank.provider | max_documents=100 |

| 处理器 | 公开成员 | 说明 |
| --- | --- | --- |
| EmbedderHandler | provider、encode(texts) | 分批编码并按 embedding.dim 逐条校验 |
| LLMHandler | provider、chat(messages)、complete(prompt, system) | 单轮对话与单条生成 |
| RerankHandler | provider、rerank(query, documents, top_k) | 透传给 provider |

`reranks/qwen_rerank.py`(153) 直接 requests.post 调 DashScope 原生 text-rerank，失败时按原顺序返回并给 0.5 中立分。

### 4.4 门面层（memories.py，344 行）

模块级只剩四个绑定（logger、__all__、REQUIRED_PARTITIONS、HandlerType）与 11 个私有函数，公开面是三个类。

| 入口 | 类属性 | 产物 | 配置要求 |
| --- | --- | --- | --- |
| `Memories` | is_async=False、handler_cls=MemoryHandler | MemoryHandler | db.backend 与 vector.backend 取同步值 |
| `AsyncMemories` | is_async=True、handler_cls=AsyncMemoryHandler | AsyncMemoryHandler | 必须取 _async 值 |

| 方法 | 签名 | 职责 |
| --- | --- | --- |
| `load` | `(json_text=None, yaml_path=None, use_env=True, env_prefix=LOONG_MEMORIES__)` | 装载配置后转 create |
| `create` | `(settings: MemorySettings)` | 只吃配置，不接受组件注入 |

| 模块级私有函数 | 职责 |
| --- | --- |
| `_load_settings` | 两个入口共用的 load_config 包装 |
| `_db_is_async` | isinstance(db, AsyncDBEngine) |
| `_engine_is_async` | isinstance(vector, AsyncVectorEngine) |
| `_store_is_async` | 底层向量库实现的 search 是否为协程 |
| `_check_family` | 族不一致时抛 ConfigError，文案带实际类名 |
| `_check_settings` | 总开关与四个硬依赖分区门控 |
| `_db_kwargs` | 按 backend 前缀组装连接池与三方言参数 |
| `_vector_kwargs` | 按 backend 前缀组装向量库参数，空串过滤 |
| `_store_class` | 归一 _async 后缀后解析实现类，异步入口找协程子类 |
| `_build_vector` | 组装引擎并注入 collection 与 vector_size |
| `_assemble` | 六分区构造 + 三层族校验 + 构造 handler |

### 4.5 领域层与流程编排（memory）

| 文件 | 内容 |
| --- | --- |
| `base.py`(96) | MemoryBase 抽象：get、page_query、add、search、update、delete；tenant_id 位置必填、user_id 位置可选 |
| `models.py`(96) | MemoryItem（含 created_at_tz、updated_at_tz）与 AddResult、GetResult、PageQueryResult、SearchResult、UpdateResult、DeleteResult |
| `scope.py`(135) | validate_entity_id、update_metadata_keys、build_new_metadata、escape_scope_value、session_scope、validate_search_params、validate_query，全为纯函数 |
| `prompts.py`(617) | ADDITIVE_EXTRACTION_PROMPT、AGENT_CONTEXT_SUFFIX、PAST_MESSAGE_TRUNCATION_LIMIT 及 6 个渲染辅助 |
| `stores.py`(612) | CORE_KEYS、keyword_text、to_vector_id_int、build_payload、split_payload，MemoryStore 与 AsyncMemoryStore |
| `handler.py`(816) | DEFAULT_THRESHOLD=0.1、DEFAULT_TOP_K=20、DEFAULT_CONTEXT_TOP_K=10、DEFAULT_KEYWORD_WEIGHT=0.5、DEFAULT_HISTORY_MESSAGES=10；MemoryHandler 与 AsyncMemoryHandler |

#### payload 键名两套并存

| 生产者 | 写入的键 |
| --- | --- |
| `stores.build_payload` | text、text_lemma、hash、created_at_tz、updated_at_tz，外加 tenant_id / user_id / agent_id |
| `stores.CORE_KEYS` | id、tenant_id、user_id、text、**text_lemmatized**、hash、created_at_tz、updated_at_tz |
| 向量库实现实际读取 | text、text_lemma、created_at_tz、user_id、tenant_id |

`CORE_KEYS` 写的是 `text_lemmatized`，生产者产出的是 `text_lemma`，两套拼写并存；任何按 `CORE_KEYS` 取词项列的代码都会拿到 None，更新链路会静默把正文与词项写空。

#### 存储封装方法签名（AST 实测）

| 方法 | 签名 |
| --- | --- |
| get_memory | (memory_id, tenant_id, user_id=None) |
| update_memory | (memory_id, tenant_id, vector, payload, previous, user_id=None) |
| delete_memory | (memory_id, tenant_id, previous, payload, user_id=None) |
| load_history | (session_scope, tenant_id, limit, user_id=None) |
| recall | (query, tenant_id, user_id=None, *, vector, filter, top_k, threshold=0.1, explain=False) |
| insert_memories | (entries, session_scope, messages=None) —— 租户与用户从 entry 顶层读取 |
| build_payload | (text, new_metadata, created_at=None) —— nlp_setting.enable 为 True 才做词形还原 |

#### 引擎封装方法签名（AST 实测）

| 方法 | 签名 |
| --- | --- |
| insert | (vectors, payloads, ids) —— 无 tenant_id，租户只经 payload 表达 |
| update | (vector_id, tenant_id, user_id=None, vector=None, payload=None) |
| delete | (vector_id, tenant_id, user_id=None) |
| search | (query, vectors, tenant_id, user_id=None, top_k=None, filter=None) |
| keyword_search | (query, tenant_id, user_id=None, top_k=None, filter=None) |
| get | (vector_id, tenant_id, user_id=None) |
| list | (tenant_id, user_id=None, filter=None, top_k=None) |

#### 双路召回与打分

```text
recall()
├── vector_engine.search(query, vector, tenant_id, user_id, internal_limit, filter)   语义路，内部召回 max(top_k*4, 60)
├── supports_keyword 判定后 keyword_search(lemmatized, tenant_id, user_id, top_k, filter) 到 normalize_bm25
├── score_and_rank(semantic_results, bm25_scores, entity_boosts, threshold, top_k, explain)
└── _build_recall_result(explain, scored)   payload 拆核心与 metadata，explain 时附 score_details
```

`supports_keyword` 的判定是 `type(store).keyword_search is not VectorStoreBase.keyword_search`，且 `nlp_setting` 为空或 `enable=False` 时直接返回 False。

## 5. 存储与基础设施层

### 5.1 向量库（storage/memories）

| 文件 | 内容 |
| --- | --- |
| `base.py`(188) | VectorStoreBase 八个抽象方法；keyword_search 默认返回 None；VectorStoreResult 字段为 id、tenant_id、user_id、text、text_lemma、updated_at_tz、created_at_tz、score、metadata；METADATA_REMOVE_KEYS 与 SESSION_PARAMS 定义在此 |
| `registry_vector.py`(63) | BUILTIN_STORES 四值（milvus、milvus_async、pg、pg_async），提供 register 与 resolve |
| `vector_engine.py`(326) | VectorEngine 与 AsyncVectorEngine；模块级只剩 build_vector_store 与 create_vector_engine |
| `milvus_store.py`(568) | 单 schema：id、tenant_id、user_id、vectors、metadata、text、text_lemma、sparse、两个时间列；BM25 Function 由 text_lemma 生成 sparse |
| `pg_store.py`(945) | 表 vut_vectors；余弦按距离归一转相似度；关键词用 to_tsvector 与 ts_rank_cd；metadata 过滤走 JSONB 包含操作符 |

VectorStoreResult 是只读视图，**不含 payload 字段**：上层必须由 text、text_lemma 与 metadata 重新拼装，或用 handler 的 `_to_payload()`。

构造期守卫（实测）：`VectorEngine.__init__` 里对异步后端的拒绝逻辑仍是注释状态（`# if "async" in store_backend: raise ...`），因此同步引擎可以包住异步方言；混用由门面在装配期拦下。`AsyncVectorEngine.__init__` 用 `if "async" not in store_backend` 拒绝同步实现，若 backend 为 None 会抛 TypeError 而非领域异常。`__enter__` 与 `__exit__` 已移除，引擎不再支持 with 语句。

### 5.2 关系库（storage/message）

| 文件 | 内容 |
| --- | --- |
| `schema.py`(120) | get_history_table 与 get_messages_table（支持 table_prefix）；两表均含 tenant_id NOT NULL、user_id、created_at_tz；HistoryRecord 与 Message 两个 pydantic 模型 |
| `db_engine.py`(402) | DBEngine 与 AsyncDBEngine：_init_db、_ensure_schema、add_history、batch_add_history、get_history、save_messages、get_last_messages |
| `sqlite_db.py`(249) | SQLiteDB（backend=sqlite）与 AsyncSQLiteDB（backend=sqlite_async） |
| `mysql_db.py`(352) | MySQLDB 与 AsyncMySQLDB，驱动在 PyMySQL 与 mysql-connector 间回退，异步在 aiomysql 与 asyncmy 间回退；sql_mode 为空串时不再拼 init_command |
| `pg_db.py`(342) | PGDB 与 AsyncPGDB，驱动在 psycopg3 与 psycopg2 间回退；sslmode 为空时回落 disable |
| `registry_db.py`(57) | BUILTIN_DBS 六值，提供 resolve_db；门面直接使用 |

**`build_database` 与 `_sqlite_path` 已从 db_engine.py 移除**，关系库实例化改由门面 `memories._db_kwargs` 加 `resolve_db` 完成。消息写入按「租户 + 会话」分组淘汰，只保留每会话最近 keep_last 条；历史写入失败由上层降级为逐条重试并告警。

## 6. 端到端核心数据流

```text
装配    config（YAML 或 JSON 或 env）→ load_config → MemorySettings
        → _check_settings 门控 → 六分区构造 → 三层族校验 → MemoryHandler 或 AsyncMemoryHandler
        → 调用方自行 start() 或 await start()

add     messages → build_new_metadata → _recall_existing（嵌入 + 双路召回）→ _extract_facts（LLM）
        → memory_hash 去重 → 批量嵌入 → entries[{id, vector, payload, tenant_id, user_id}] → insert_memories

search  query → 嵌入 → recall（语义 + 关键词 + score_and_rank）→ 可选 rerank → SearchResult

get     memory_id + tenant_id（+ user_id）→ 向量库读取 → _to_payload → MemoryItem → GetResult.result

page_query  tenant_id（+ user_id）→ 向量库 list 抓 (page_num+1)*page_size → 切片 → PageQueryResult

update  get → _update_payload → 正文变化才重算向量 → update_memory（向量库 + 历史 UPDATE）

delete  get → delete_memory（向量库物理删除 + 历史 DELETE）
```

| 阶段 | 关键校验 | 失败策略 |
| --- | --- | --- |
| 装配门控 | 总开关 + 四个硬依赖分区 + 三层实现族 | 抛 ConfigError，构造前即失败 |
| 作用域构造 | metadata 身份键剥离；agent_id 非法即报错 | 抛 DataError |
| 嵌入 | embedding.dim 逐条校验 | 抛 EmbeddingError |
| 事实抽取 | 模型响应按 JSON 解析，兼容代码块包裹 | 抛 LLMError；非 JSON 视为未抽取 |
| 向量写入 | 维度与批量长度校验 | 失败抛 StorageError |
| 历史与消息 | 无 | 只告警，不回滚主写入 |
| 重排 | 文档数不超过 max_documents | 异常降级为原顺序，给 0.5 中立分 |

## 7. 配置与环境变量

```bash
LOONG_MEMORIES__ENABLE=true
LOONG_MEMORIES__EMBEDDING__PROVIDER=qwen
LOONG_MEMORIES__EMBEDDING__DIM=1024
LOONG_MEMORIES__LLM__PROVIDER=qwen
LOONG_MEMORIES__DB__BACKEND=sqlite
LOONG_MEMORIES__DB__SQLITE__DB_PATH=data/mem.db
LOONG_MEMORIES__VECTOR__BACKEND=pg
LOONG_MEMORIES__VECTOR__COLLECTION=memories
LOONG_MEMORIES__VECTOR__PG__DATABASE=loong
LOONG_MEMORIES__NLP__ENABLE=true
```

- 前缀固定 LOONG_MEMORIES__，层级用双下划线；值默认按字符串交给 pydantic 转换，仅以方括号或花括号开头时按 JSON 解析。
- load_config 的 yaml_path 与 json_text 可选，叠加顺序 YAML、JSON、环境变量（后者最高）。
- MemorySettings 关闭了 pydantic-settings 的原生 env、dotenv 与 secrets 来源，环境变量只有 loader_config 一个入口。
- `db.backend` 六个取值：sqlite、mysql、pg、sqlite_async、mysql_async、pg_async；`vector.backend` 四个取值：milvus、pg、milvus_async、pg_async。
- 同步入口配无后缀值，异步入口配 `_async` 值；门面按实现类判定族，混用在装配期报错。

## 8. 测试

| 文件 | 用例数 | 行数 | 覆盖范围 |
| --- | --- | --- | --- |
| conftest.py | - | 245 | 路径注入、两个 CLI 选项、三标记注册、六份配置字典与 Settings 对象 |
| test_loader_config.py | 84 | 513 | load_config 三级来源、深度合并、环境变量解析、异常分支 |
| test_embedding.py | 35 | 283 | 注册表、内置 embedder、EmbedderHandler 批次与维度校验 |
| test_llm.py | 38 | 299 | 注册表、内置 provider、LLMHandler |
| test_sqlite_db.py | 50 | 521 | 关系库契约与业务矩阵，默认可跑 |
| test_integration_vector_engine.py | 51 | 605 | 引擎契约与后端矩阵 |
| test_integration_stores.py | 45 | 992 | MemoryStore 与 AsyncMemoryStore 读写、审计与召回 |
| test_integration_pg_store.py | 56 | 785 | PG 向量库契约与业务矩阵 |
| test_integration_milvus_store.py | 58 | 713 | Milvus 向量库契约与业务矩阵 |
| test_integration_mysql_db.py | 49 | 613 | MySQL 关系库契约与业务矩阵 |
| test_integration_pg_db.py | 49 | 603 | PG 关系库契约与业务矩阵 |
| test_integration_ai_memory_handler.py | 36 | 879 | 处理器端到端（同步与异步各一组） |
| test_integration_ai_memories.py | 68 | 947 | 门面端到端：契约 59 项（含 inspect.signature 比对与三层族判定）+ 真实库 9 项 |

运行方式：addopts 为 `-s -q -rs`，`-m integration` 已注释；`--integration` 与 `--integration_ai` 是显式开关。集成文件在模块顶部做参数门控，缺依赖或缺实例时整模块 skip。

本地环境缺 sqlalchemy、pydantic_settings、dotenv、pytest_asyncio 时，`memories` 与 `memory.handler` 相关文件在 import 阶段整体 skip，静态结论不等于运行期通过。

## 9. 已知缺陷索引

本文件只描述结构。缺陷清单与修复补丁见 `.vcl/defect_report.md`，当前仍成立的高优先级项：

| # | 位置 | 现象 |
| --- | --- | --- |
| 1 | `stores.CORE_KEYS` | 写的是 text_lemmatized，生产者是 text_lemma，按它取词项列会拿到 None |
| 2 | `pg_store._build_sql_insert` | 批量长度校验用 `and` 连接两个不等式，只有三处长度全错才报错 |
| 3 | `vector_engine.VectorEngine.__init__` | 异步后端拒绝逻辑仍是注释，同步引擎可包住异步方言 |
| 4 | `vector_engine.AsyncVectorEngine.__init__` | backend 为 None 时 `in` 判定抛 TypeError 而非领域异常 |
| 5 | `vector_engine.build_vector_store` | 不查注册表，新增向量后端必须同时改 if 分支 |
| 6 | `scope.build_new_metadata` | 三个实体 ID 至少其一的校验被注释，租户硬边界完全依赖 handler 参数必填 |
| 7 | `core.config_vector` | VectorSettings.backend 的 Literal 里 milvus 重复出现 |
| 8 | `memories._store_class` | 依赖「异步实现与同步实现同模块」这一约定，跨模块注册的自定义异步实现会命中 ConfigError |

## 10. 扩展落点

| 需求 | 落点 | 注意 |
| --- | --- | --- |
| 新增向量库后端 | 继承 VectorStoreBase 并声明 backend；同时在 config_vector 的 Literal 与 build_vector_store 分支放行 | 注册表已自动登记，但 build_vector_store 不查注册表；异步版需与同步版同模块且 search 为协程 |
| 新增模型厂商 | 继承 BaseEmbedder、BaseLLMProvider 或 BaseRerankProvider，覆盖 backend、default_model、base_url、api_key_env，必要时覆盖 max_batch_size | provider 是自由字符串，无需改 Literal |
| 新增关系库后端 | 继承 BaseDB，提供 start 与 _missing_tables，并在 config_db 的 Literal 放行 | 门面经 resolve_db 解析，注册表即生效；同步与异步实现需分别登记 |
| 调整打分策略 | `util/scoring.py` 的 score_and_rank 与 get_bm25_params | 权重与阈值常量在 `memory/handler.py` 顶部 |
| 新增转换工具 | `util/numberic_util.py`、`util/datetime_util.py` | 存储层已统一经 str_to_int 与 ensure_datetime 归一 |
| 新增配置分区 | `core/config_memory.py` 加子模型并在 MemorySettings 声明 default_factory | 同步 conftest 的 CONFIG 字典、README 配置表与门面的分区构造 |
| 对话历史滚动压缩 | 门控与裁剪进 `memory/handler.py`，摘要复用 LLMHandler，阈值进 `core/config_memory.py` | 摘要识别与轮次切分宜做纯函数模块，避免被 splitIntoTurns 计入轮次 |

## 11. 本次扫描结论

- 分层骨架完整且收敛：core 配置（六分区）、能力层三套同构注册表、memory 编排、storage 双方言双同步异步，职责边界清晰。
- 装配入口补齐：`memories.py` 用「基类放 load 与 create、子类只声明两个类属性」的形态收敛了此前手工装配流程，并把三层实现族校验前移到装配期。
- 实例化责任上移：`db_engine.build_database`、`vector_engine.build_vector_engine`、`vector_engine._backend_kwargs` 均已移除，连接参数组装统一在门面；关系库走注册表，向量库仍受 if 分支限制。
- 主要剩余风险是 payload 键名两套拼写（text_lemma 与 text_lemmatized）与向量库侧的空正文写回，属静默数据损坏，优先级高于其余问题。
- `vector.backend` 的 Literal 重复值、`AsyncVectorEngine` 对 None backend 的 TypeError、`_store_class` 的同模块假设，是门面引入后的三个新边界点。
- 测试已按新契约重写：门面文件 68 项、处理器文件 36 项，集成用例按后端矩阵参数化，但本地缺 sqlalchemy 与 pydantic-settings，规则化回归仍需在完整 hatch 环境中执行。
