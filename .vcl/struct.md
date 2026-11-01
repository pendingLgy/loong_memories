# loong_memories 项目结构与架构分析

> 依据：本地源码静态扫描（目录遍历 + 行数统计 + 模块头注释与公共签名阅读），不含运行时推断。
> 环境：Windows 11 (10.0.22631)、CPython 3.13.13、cmd.exe、代码页 936；源码 UTF-8 无 BOM、CRLF。
> 时点：git HEAD `a518ee6`（调整参数）；工作区另有未提交改动（`memory/handler.py` 修改中、新增 `test_integration_ai_memory_handler.py`、未跟踪 `test.db`）。

## 1. 规模与基本信息

### 1.1 源码规模

| 范围 | 文件数 | 行数 | 备注 |
| --- | --- | --- | --- |
| `src/loong_memories` | 60 个 `.py` | 8566 | 含 Apache 许可头与 docstring |
| `tests` | 12 个 `.py` | 6814 | 静态 `def test_` 计数 528 项，未展开参数化 |
| 仓库根 | pyproject.toml、README.md、DEVELOPMENT.md、LICENSE | - | 文档另计 |

最大八个文件：`pg_store.py`(943)、`handler.py`(636)、`prompts.py`(617)、`stores.py`(584)、`milvus_store.py`(568)、`db_engine.py`(466)、`vector_engine.py`(363)、`mysql_db.py`(352)。

唯一位 0 行文件：`reranks/__init__.py`。仅 `embedding/__init__.py`(41) 有导出（9 个符号加 `BUILTIN_EMBEDDERS`）；`llm`、`memory`、`nlp`、`storage` 子包、`util`、`core` 的 `__init__.py` 只有许可头。包根 `__init__.py`(13) 与门面 `memories.py`(13) 同样只有许可头，未导出任何符号。

`util` 已从 3 文件扩为 6 文件：新增 `str_util.py`(6)、`numberic_util.py`(23)、`datetime_util.py`(30)，`memory_hash` 与时间、数值转换从 `stores.py`、`pg_store.py` 的内联实现上收为公共工具。

目录内残留 `__pycache__` 下的 `cpython-312` 字节码，与当前解释器 3.13 不一致，属历史构建产物。

### 1.2 分发与构建

| 项 | 值 |
| --- | --- |
| 分发名 | loong_memories（描述：llm long memory） |
| 版本 | 0.1.0，由 `src/loong_memories/__about__.py` 动态读取 |
| 构建后端 | hatchling，src-layout |
| 声明 Python | `>=3.8`（实际运行 3.13.13，代码已用 3.9+ 的 `dict[str, Any]` 与 `X | None` 语法） |
| 许可 | Apache-2.0 |
| 测试配置 | testpaths=tests、asyncio_mode=auto、asyncio_default_fixture_loop_scope=module、addopts=`-s -q -rs`、开启 log_cli |
| 自定义标记 | unit、integration（conftest 的 pytest_configure 注册）；另保留 CLI 选项 `--integration`、`--eval` |

`addopts` 中的 `-m integration` 已被注释，集成标记不再参与反选，集成用例是否能跑由 conftest 内的连接探测决定。

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

`reranks/qwen_rerank.py` 直接用 `requests.post` 调 DashScope，`requests` 已列入运行时依赖。

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
│     ├─ __init__.py(13)                     仅许可头，无导出
│     ├─ __about__.py(16)                    版本号 0.1.0，供 hatch 动态读取
│     ├─ memories.py(13)                     顶层门面，仅许可头，装配逻辑未实现
│     ├─ registry.py(61)                     通用 Registry 与 Ref 类型别名
│     ├─ commons/
│     │  └─ __init__.py(13)                  空包，仅许可头
│     ├─ core/
│     │  ├─ __init__.py(13)                  仅许可头
│     │  ├─ config_db.py(140)                DBSettings、SQLiteSettings、MySQLSettings、PGSettings、DBPoolSettings
│     │  ├─ config_llm.py(106)               EmbeddingSettings、LLMSettings、RerankSettings
│     │  ├─ config_memory.py(78)             MemorySettings 总配置，关闭原生 env 来源，校验维度一致
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
│     │  ├─ base.py(89)                      MemoryBase 抽象：get、page_query、add、search、update、delete
│     │  ├─ handler.py(636)                  流程编排：MemoryHandler 与 AsyncMemoryHandler
│     │  ├─ models.py(96)                    MemoryItem 与六种结果对象（时间字段已改名 _tz 后缀）
│     │  ├─ prompts.py(617)                  抽取提示词常量与渲染函数
│     │  ├─ scope.py(145)                   作用域构造、会话键、参数校验，全为纯函数
│     │  └─ stores.py(584)                   MemoryStore 与 AsyncMemoryStore，双路召回与打分
│     ├─ nlp/
│     │  ├─ __init__.py(13)                  仅许可头
│     │  ├─ lemmatization.py(163)            中英文词形还原，供 BM25 使用
│     │  └─ spacy.py(115)                    spaCy 管线双缓存、失败哨兵与句子边界
│     ├─ reranks/
│     │  ├─ __init__.py(0)                   空文件，无任何内容
│     │  ├─ base.py(98)                      BaseRerankProvider 抽象
│     │  ├─ qwen_rerank.py(153)              DashScope 原生 text-rerank 调用与失败降级
│     │  ├─ registry_rerank.py(61)           重排实现注册表与解析函数
│     │  └─ rerank_handler.py(64)            RerankHandler：按配置装配并透传调用
│     ├─ storage/
│     │  ├─ memories/
│     │  │  ├─ __init__.py(13)               仅许可头
│     │  │  ├─ base.py(182)                  VectorStoreBase 八个抽象方法、VectorStoreResult 与结果转换
│     │  │  ├─ milvus_store.py(568)           MilvusVectorStore 与 AsyncMilvusVectorStore
│     │  │  ├─ pg_store.py(943)               PGVectorStore 与 AsyncPGVectorStore，表 vut_vectors
│     │  │  ├─ registry_vector.py(63)         向量库注册表，仅写入未被读取
│     │  │  └─ vector_engine.py(363)         VectorEngine 与 AsyncVectorEngine，构造后端并校验入参
│     │  └─ message/
│     │     ├─ __init__.py(13)               仅许可头
│     │     ├─ base.py(67)                   BaseDB 抽象，声明 backend 后自动登记
│     │     ├─ db_engine.py(466)             DBEngine 与 AsyncDBEngine，通用 CRUD 与建表
│     │     ├─ mysql_db.py(352)              MySQLDB 与 AsyncMySQLDB，驱动双回退
│     │     ├─ pg_db.py(341)                 PGDB 与 AsyncPGDB，驱动在 psycopg3 与 psycopg2 间回退
│     │     ├─ registry_db.py(57)            关系库注册表，仅写入未被读取
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
   ├─ conftest.py(230)                     路径注入、命令行选项、标记注册、配置拼装
   ├─ test_embedding.py(283)               注册表、内置 embedder 与 handler，35 项
   ├─ test_llm.py(299)                     注册表、内置 provider 与 handler，38 项
   ├─ test_loader_config.py(513)           配置三级来源与异常分支，84 项
   ├─ test_integration_ai_memory_handler.py(660)   处理器端到端，同步与异步各一组，12 项
   ├─ test_integration_milvus_store.py(713)        Milvus 契约与业务矩阵，58 项
   ├─ test_integration_mysql_db.py(613)           MySQL 契约与业务矩阵，49 项
   ├─ test_integration_pg_db.py(603)              PG 关系库契约与业务矩阵，49 项
   ├─ test_integration_pg_store.py(785)           PG 向量库契约与业务矩阵，56 项
   ├─ test_integration_stores.py(981)             MemoryStore 与 AsyncMemoryStore 读写、审计与召回，45 项
   ├─ test_integration_vector_engine.py(613)      引擎契约与后端矩阵，52 项
   └─ test_sqlite_db.py(521)                      SQLite 契约与业务矩阵，50 项
```

### 目录规模汇总

| 目录 | 文件数 | 行数 |
| --- | ---: | ---: |
| src/loong_memories/core | 7 | 683 |
| src/loong_memories/embedding | 6 | 417 |
| src/loong_memories/llm | 7 | 399 |
| src/loong_memories/memory | 7 | 2180 |
| src/loong_memories/nlp | 3 | 291 |
| src/loong_memories/reranks | 5 | 376 |
| src/loong_memories/storage/memories | 6 | 2132 |
| src/loong_memories/storage/message | 8 | 1665 |
| src/loong_memories/util | 6 | 307 |
| src/loong_memories/commons | 1 | 13 |
| 包根（init、about、memories、registry） | 4 | 103 |
| 合计 | 60 | 8566 |
## 3. 分层与依赖方向

```text
memory/handler        流程编排：get、page_query、add、search、update、delete
   ├── memory/stores          MemoryStore / AsyncMemoryStore：向量库与关系库组合封装
   │      ├── storage/memories/vector_engine   VectorEngine / AsyncVectorEngine
   │      │      └── storage/memories/(milvus_store | pg_store)
   │      └── storage/message/db_engine        DBEngine / AsyncDBEngine
   │             └── storage/message/(sqlite_db | mysql_db | pg_db)
   ├── memory/scope        纯函数：作用域构造、会话键、参数校验
   ├── memory/prompts      提示词常量与渲染
   ├── embedding/embedder_handler   EmbedderHandler
   ├── llm/llm_handler              LLMHandler
   └── reranks/rerank_handler       RerankHandler

core/  只承载配置、异常、装载与注册表，不依赖任何领域服务。
util/scoring 与 util/(str_util | numberic_util | datetime_util) 及 nlp/ 为纯函数层，被 memory 与 storage 复用。
storage/ 的方言实现全部在函数内惰性导入，缺驱动不影响包导入。
```

### 3.1 解耦手段

| 手段 | 落点 | 效果 |
| --- | --- | --- |
| 自动注册表 | `VectorStoreBase.__init_subclass__`、`BaseDB.__init_subclass__` | 子类声明 backend 即登记，无需显式注册调用 |
| 惰性导入 | `vector_engine.build_vector_store`、`db_engine.build_database`、各方言模块 | 缺 pymilvus 或 psycopg 时仍可 import 包 |
| 抽象基类 | MemoryBase、VectorStoreBase、BaseDB、BaseEmbedder、BaseLLMProvider、BaseRerankProvider | 上层只依赖契约 |
| 配置驱动 | `core.loader_config` 加 pydantic 模型 | YAML、JSON、env 三层深合并 |
| 公共工具 | `util.str_util`、`util.numberic_util`、`util.datetime_util` | 指纹、ID 归一与时间转换逻辑单点维护 |

已知不一致：`build_vector_store` 与 `build_database` 都用 if 分支直连具体类，未查询注册表（`registry_vector`、`registry_db` 只写不读），新增后端必须同时改 Literal 与分支。门面 `memories.py` 仍为空，装配入口缺失。

## 4. 核心模块职责拆解

### 4.1 配置与初始化层（core）

| 模型 | 关键字段（默认值） |
| --- | --- |
| EmbeddingSettings | enable(False)、provider(openai)、model、batch_size(128)、dim(1024)、api_key、base_url、timeout(60，必须大于 0) |
| LLMSettings | enable(False)、provider、model、temperature(0.7，0 到 2)、max_tokens(0 表示不传)、api_key、base_url、timeout(60) |
| RerankSettings | enable(False)、provider(qwen)、model、top_k(5)、max_tokens(0)、max_documents(100)、api_key、base_url、timeout(60) |
| MemorySettings | extra=forbid；embedding、llm、rerank、db、vector 五个子配置 default_factory；settings_customise_sources 只保留 init_settings；校验 embedding.dim 与 vector.vector_size 一致 |
| DBSettings | enable、backend（Literal：sqlite、mysql、pg 及三个 _async 变体）、sqlite、mysql、pg、table_prefix、pool_config |
| SQLiteSettings | db_path（默认空串，运行时校验）、timeout(30)、isolation_level、check_same_thread(False) |
| MySQLSettings | host、port、user、password、database、charset、connect_timeout、autocommit、sql_mode |
| PGSettings | host、port、user、password、database、connection_string、sslmode、connect_timeout、application_name、options、target_session_attrs |
| DBPoolSettings | print_sql、pool_size(5)、max_overflow(10)、pool_timeout(5)、pool_recycle(1800)、pool_pre_ping |
| VectorSettings | enable、backend（Literal：milvus、pg）、collection、vector_size(1024)、top_k(5)、metric_type、milvus、pg、pool_config |
| MilvusSettings | uri、token、db_name |
| PGVectorSettings | connection_string、host、port、user、password、database、use_hnsw(True)、use_diskann(False)、sslmode 等高级参数 |

装载入口 `load_config(json_text=None, yaml_path=None, use_env=True, env_prefix=LOONG_MEMORIES__)`：YAML 到 JSON 到环境变量依次深合并，再交给 `MemorySettings.model_validate`，失败统一抛 ConfigError。辅助函数：load_config_from_json、load_config_from_yaml、load_config_from_env、_deep_merge、_ensure_mapping、_set_nested、_coerce_env_value、_import_yaml。

异常层次（`core/exceptions.py`，50 行）：RagDataError 派生出 ConfigError、DataError、OptionalDependencyError（再到 ParserDependencyError）、ParserError、EmbeddingError、LLMError、StorageError。

### 4.2 工具与底层支撑层（util、nlp、registry）

| 文件 | 内容 |
| --- | --- |
| `util/snowflake.py` | EPOCH_MS=1704067200000；SnowflakeGenerator(datacenter_id, worker_id) 加锁生成；模块级 next_id() 用进程内单例 |
| `util/str_util.py` | memory_hash(text)：MD5 指纹，用于 add 去重与 update 变更判定 |
| `util/numberic_util.py` | str_to_int 与 str_to_float：空值与空串返回 None，解析失败抛 ParserError；供存储层把字符串实体 ID 归一为整型 |
| `util/datetime_util.py` | ensure_datetime、now_datetime、now_iso、datetime_to_isoformat：统一时间列写入与 ISO 序列化 |
| `util/scoring.py` | get_bm25_params(query, lemmatized=) 按词数选 sigmoid 参数；normalize_bm25；ENTITY_BOOST_WEIGHT=0.5；score_and_rank 先按 threshold 门控语义分，再除以动态上限 |
| `nlp/spacy.py` | DEFAULT_MODEL=zh_core_web_md；_PIPELINES 与 _PIPELINES_LEMMA 双缓存共用一把 Lock，失败写哨兵 _FAILED；load_nlp、get_nlp_lemma、clear_cache、ensure_sentence_boundaries |
| `nlp/lemmatization.py` | lemmatize_for_bm25、lemmatize_zh_for_bm25（中文与复合标识符，保留下划线与连字符） |
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

`reranks/qwen_rerank.py`(153) 直接 `requests.post` 调 DashScope 原生 text-rerank，失败时按原顺序返回并给 0.5 中立分。

### 4.4 领域层与流程编排（memory）

| 文件 | 内容 |
| --- | --- |
| `base.py` | MemoryBase 抽象：get、page_query、add、search、update、delete 六个方法 |
| `models.py` | MemoryItem（id、memory、score、hash、created_at_tz、updated_at_tz、user_id、agent_id、tenant_id、actor_id、role、metadata、score_details）与 AddResult、GetResult、PageQueryResult、SearchResult、UpdateResult、DeleteResult |
| `scope.py` | ENTITY_PARAMS；validate_entity_id、strip_identity_keys、build_scope（返回 payload 与 query，三个实体 ID 至少其一）、escape_scope_value、session_scope、validate_search_params、validate_query |
| `prompts.py` | ADDITIVE_EXTRACTION_PROMPT、AGENT_CONTEXT_SUFFIX、PAST_MESSAGE_TRUNCATION_LIMIT、generate_additive_extraction_prompt 及 6 个渲染辅助函数 |
| `stores.py` | CORE_KEYS、keyword_text、to_vector_id_int、build_payload、split_payload，MemoryStore 与 AsyncMemoryStore |
| `handler.py` | 常量 DEFAULT_THRESHOLD=0.1、DEFAULT_TOP_K=20、DEFAULT_CONTEXT_TOP_K=10、DEFAULT_KEYWORD_WEIGHT=0.5、DEFAULT_HISTORY_MESSAGES=10；MemoryHandler 与 AsyncMemoryHandler |

#### payload 键名并存（当前最大的一致性风险）

| 生产者 | 写入的键 |
| --- | --- |
| `stores.build_payload` | text、text_lemma、hash、created_at_tz、updated_at_tz，外加 user_id 或 agent_id 或 tenant_id |
| `handler._merge_payload` | data、hash、text_lemma、created_at、updated_at |
| `stores.CORE_KEYS` | id、tenant_id、user_id、text、text_lemmatized、hash、created_at_tz、updated_at_tz |
| 向量库实现实际读取 | text、text_lemma、created_at_tz、user_id、tenant_id |

三套键名互不覆盖，且 `CORE_KEYS` 写的是 `text_lemmatized` 而生产者产出 `text_lemma`，导致核心字段反向漏进 metadata。详见 `.vcl/defect_report.md`。

#### 存储封装方法签名（实测）

| 方法 | 签名 |
| --- | --- |
| get_memory | (memory_id, tenant_id, user_id=None) |
| update_memory | (memory_id, tenant_id, vector, payload, previous, user_id=None) |
| delete_memory | (memory_id, tenant_id, previous, payload, user_id=None) |
| load_history | (session_scope, tenant_id, limit, user_id=None) |
| recall | (query, tenant_id, vector, filter, top_k, threshold=0.1, explain=False, user_id=None) —— 过滤参数名已由 filters 改为 filter |
| insert_memories | (entries, session_scope, messages=None) —— 租户与用户从 entry 顶层读取 |

#### 引擎封装方法签名（实测，insert 已去掉租户入参）

| 方法 | 签名 |
| --- | --- |
| insert | (vectors, payloads, ids) —— 无 tenant_id，租户只经 payload 表达 |
| update | (vector_id, tenant_id, user_id=None, vector=None, payload=None) |
| delete | (vector_id, tenant_id, user_id=None) |
| search | (query, vectors, tenant_id, user_id=None, top_k=None, filter=None) |
| keyword_search | (query, tenant_id, user_id=None, top_k=None, filter=None) |
| get | (vector_id, tenant_id, user_id=None) |
| list | (tenant_id, user_id=None, filters=None, top_k=None) |

#### 双路召回与打分

```text
recall()
├── vector_engine.search(query, vector, tenant_id, user_id, internal_limit, filter)  语义路，内部召回 max(top_k*4, 60)
├── supports_keyword 判定后 keyword_search(lemmatized, tenant_id, user_id, top_k, filter) 到 normalize_bm25
├── score_and_rank(semantic_results, bm25_scores, entity_boosts, threshold, top_k, explain)
└── _build_recall_result(explain, scored)   payload 拆核心与 metadata，explain 时附 score_details
```

`keyword_text` 已并入 `stores.py`，`memory_hash` 与时间工具已上移到 `util` 包。

## 5. 存储与基础设施层

### 5.1 向量库（storage/memories）

| 文件 | 内容 |
| --- | --- |
| `base.py`(182) | VectorStoreBase 八个抽象方法（start、create_col、insert、search、delete、update、get、list）与 keyword_search 默认返回 None；VectorStoreResult 字段为 id、tenant_id、user_id、text、text_lemma、updated_at_tz、created_at_tz、score、metadata；to_vector_store_result 统一归一 |
| `registry_vector.py`(63) | 纯注册表：register_vector_store、available_vector_stores、resolve_vector_store |
| `vector_engine.py`(363) | VectorEngine 与 AsyncVectorEngine（校验维度与批量长度）；build_vector_store 按 if 分支直连 milvus 与 pg，未走注册表；create_vector_engine、build_vector_engine、_backend_kwargs |
| `milvus_store.py`(568) | 单 schema：id、tenant_id、user_id、vectors、metadata(JSON)、text、text_lemma、sparse、两个时间列；BM25 Function 由 text_lemma 生成 sparse；vector_index（AUTOINDEX，COSINE）与 sparse_index（BM25）；AsyncMilvusVectorStore 以 async 前缀标记 backend |
| `pg_store.py`(943) | 表 vut_vectors：id、tenant_id、user_id、vector、text、text_lemma、metadata(JSONB)、两个时间列；余弦按距离归一转相似度；关键词用 to_tsvector(simple, text_lemma) 与 ts_rank_cd；metadata 过滤走 JSONB 包含操作符 |

VectorStoreResult 是只读视图，**不含 payload 字段**：上层若按 dict 语义读取正文会失败，需由 text、text_lemma 与 metadata 重新拼装。

构造期守卫（当前实测）：VectorEngine 对 async 后端的拒绝逻辑已被注释（vector_engine.py 第 69 到 70 行），因此同步引擎可包裹异步方言实现；AsyncVectorEngine 自行校验后端名必须含 async。`__enter__` 与 `__exit__` 已移除，引擎不再支持 with 语句。

`store_backend = store_base.backend` 为遗留未使用局部变量；`backend` 为 None 时 `in` 判定会抛 TypeError。

### 5.2 关系库（storage/message）

| 文件 | 内容 |
| --- | --- |
| `schema.py`(120) | get_history_table 与 get_messages_table（支持 table_prefix）；两表均含 tenant_id NOT NULL、user_id、created_at_tz；历史表另有 is_deleted、actor_id、role，并建有三组租户维度索引；HistoryRecord 与 Message 两个 pydantic 模型 |
| `db_engine.py`(466) | DBEngine（_init_db、_ensure_schema、add_history、batch_add_history、get_history、save_messages、get_last_messages、_build_evict_stmt、_record_row、_message_row）与 AsyncDBEngine；build_database 按 backend 分派；_sqlite_path 取配置路径 |
| `sqlite_db.py`(249) | SQLiteDB（backend=sqlite）与 AsyncSQLiteDB（backend=sqlite_async） |
| `mysql_db.py`(352) | MySQLDB 与 AsyncMySQLDB，驱动在 PyMySQL 与 mysql-connector 间回退，异步在 aiomysql 与 asyncmy 间回退 |
| `pg_db.py`(341) | PGDB 与 AsyncPGDB，驱动在 psycopg3 与 psycopg2 间回退，异步用 asyncpg |
| `registry_db.py`(57) | 纯注册表：register_db、resolve_db |

消息写入按「租户 + 会话」分组淘汰，只保留每会话最近 keep_last 条；历史写入失败由上层降级为逐条重试并告警。get_history 与 get_last_messages 现已把 tenant_id 作为必填位置参数。

## 6. 端到端核心数据流

```text
add     messages → build_scope → _recall_existing（嵌入 + 双路召回）→ _extract_facts（LLM）
        → memory_hash 去重 → 批量嵌入 → entries[{id, vector, payload}] → insert_memories
search  query → 嵌入 → recall（语义 + 关键词 + score_and_rank）→ 可选 rerank → SearchResult
update  get_memory → _merge_payload → 正文变化才重算向量 → update_memory（向量库 + history(UPDATE)）
delete  get_memory → delete_memory（向量库物理删除 + history(DELETE)）
get | page_query   接口已声明，实现为空
```

| 阶段 | 关键校验 | 失败策略 |
| --- | --- | --- |
| 作用域构造 | 三个实体 ID 至少一个；metadata 身份键剥离 | 抛 DataError |
| 嵌入 | embedding.dim 逐条校验 | 抛 EmbeddingError |
| 事实抽取 | 模型响应按 JSON 解析，兼容代码块包裹 | 抛 LLMError；非 JSON 视为未抽取 |
| 向量写入 | 维度与批量长度校验 | 失败抛 StorageError |
| 历史与消息 | 无 | 只告警，不回滚主写入 |
| 重排 | 文档数不超过 max_documents | 异常降级为原顺序，给 0.5 分 |

实测的跨层断点（影响上述链路能否跑通，详见缺陷报告）：

1. `handler.update`、`handler.delete` 以单参数调用 `store.get_memory`，且随后访问 `current.payload`，而 `VectorStoreResult` 无该字段；
2. `handler.update` 调 `store.update_memory(memory_id, vector, payload, previous)`，与 `(memory_id, tenant_id, vector, payload, previous, user_id)` 错位且缺租户；
3. `handler._persist_facts` 产出的 entry 只有 id、vector、payload，而 `insert_memories` 从 entry 顶层取 tenant_id 与 user_id，审计记录拿到 None，撞 history 表 tenant_id NOT NULL；
4. `handler._to_message_objects` 用 created_at 关键字（模型字段已改名 created_at_tz）且不填 tenant_id，Message 恒未落库；
5. `handler._to_item` 用 created_at、updated_at 关键字构造 MemoryItem（字段为 created_at_tz、updated_at_tz），构造即抛 ValidationError；
6. `handler.get`、`handler.page_query` 只有签名与 docstring，隐式返回 None；异步侧未覆写这两个方法。

## 7. 配置与环境变量

```bash
LOONG_MEMORIES__EMBEDDING__PROVIDER=qwen
LOONG_MEMORIES__EMBEDDING__DIM=1024
LOONG_MEMORIES__LLM__PROVIDER=qwen
LOONG_MEMORIES__DB__BACKEND=sqlite
LOONG_MEMORIES__DB__SQLITE__DB_PATH=data/mem.db
LOONG_MEMORIES__VECTOR__BACKEND=pg
LOONG_MEMORIES__VECTOR__COLLECTION=memories
LOONG_MEMORIES__VECTOR__PG__DATABASE=loong
```

- 前缀固定 LOONG_MEMORIES__，层级用双下划线；值默认按字符串交给 pydantic 转换，仅以方括号或花括号开头时按 JSON 解析。
- load_config 的 yaml_path 与 json_text 可选，叠加顺序 YAML、JSON、环境变量（后者最高）。
- MemorySettings 关闭了 pydantic-settings 的原生 env、dotenv 与 secrets 来源，环境变量只有 loader_config 一个入口。
- 向量库 backend 的 Literal 只有 milvus 与 pg，异步实现无法经配置选中；数据库 backend 则包含三个 _async 变体。

## 8. 测试

| 文件 | 用例数 | 行数 | 覆盖范围 |
| --- | --- | --- | --- |
| conftest.py | - | 230 | src 路径注入、--integration 与 --eval 选项、unit 与 integration 标记注册、CONFIG_ 环境拼装、_load_memory_settings |
| test_loader_config.py | 84 | 513 | load_config 三级来源、深度合并、环境变量解析、异常分支 |
| test_embedding.py | 35 | 283 | 注册表、内置 embedder、EmbedderHandler 批次与维度校验 |
| test_llm.py | 38 | 299 | 注册表、内置 provider、LLMHandler |
| test_integration_ai_memory_handler.py | 12 | 660 | MemoryHandler 端到端（同步与异步各一组，mock 与真实存储各一组） |
| test_integration_stores.py | 45 | 981 | MemoryStore 与 AsyncMemoryStore 读写、历史审计与召回，单元与集成分离 |
| test_integration_vector_engine.py | 52 | 613 | VectorEngine 与 AsyncVectorEngine 契约与后端矩阵 |
| test_sqlite_db.py | 50 | 521 | 关系库契约与业务矩阵（默认可跑） |
| test_integration_mysql_db.py | 49 | 613 | 同上（需 MySQL 实例） |
| test_integration_pg_db.py | 49 | 603 | 同上（需 PG 实例） |
| test_integration_milvus_store.py | 58 | 713 | Milvus 向量库契约与业务矩阵 |
| test_integration_pg_store.py | 56 | 785 | PG 向量库契约与业务矩阵 |

运行方式：addopts 为 `-s -q -rs`，`-m integration` 已注释；`--integration`、`--eval` 保留为显式开关。

本轮测试文件已全面改名为 `test_integration_*` 前缀，并把 `requires_integration` 收敛到集成类级别：缺 sqlalchemy 或缺实例时只有集成类被跳过，纯函数与契约用例仍执行。本地环境未安装 sqlalchemy，依赖关系库的模块在 import 阶段整体 skip。

## 9. 已知缺陷索引

本文件只描述结构。缺陷清单、根因与修复补丁见 `.vcl/defect_report.md`，P0 集中在以下位置：

1. `memory/handler.py`：get 与 page_query 空实现；update 与 delete 以单参数调用 get_memory 后访问不存在的 payload 字段；update 调 update_memory 时参数错位且缺 tenant_id；_to_item 与 _to_message_objects 使用已改名的时间键并漏填 tenant_id。
2. `memory/stores.py`：insert_memories 从 entry 顶层取 tenant_id 与 user_id，而 handler 产出的 entry 只有 id、vector、payload；recall 中 `{**mem.model_dump(), **mem.metadata}` 在 metadata 为 None 时 TypeError；CORE_KEYS 写 text_lemmatized 而生产者产出 text_lemma。
3. `storage/memories/pg_store.py`：_build_sql_insert 的批量长度校验用 and 连接两个不等式，只在三处长度全错时才报错；_build_sql_update 中 text、text_lemma、updated_at_tz 三行落在 payload 判空之外，仅更新向量时直接 AttributeError。
4. `storage/memories/milvus_store.py`：仅更新向量时用 metadata 回填正文，而 metadata 已被剔除 text 与 text_lemma，记录会以空正文写回并从召回中消失。
5. `storage/memories/vector_engine.py`：backend 为 None 时 `in` 判定抛 TypeError；start 无返回值。
6. `memory/scope.py`：build_scope 只要求 user_id、agent_id、tenant_id 三者至少其一，而 tenant 是硬隔离边界，缺失时写入必撞 NOT NULL。

## 10. 扩展落点

| 需求 | 落点 | 注意 |
| --- | --- | --- |
| 新增向量库后端 | 继承 VectorStoreBase 并声明 backend；同时在 config_vector 的 Literal 与 build_vector_store 分支放行 | 注册表已自动登记，但 build_vector_store 不查注册表 |
| 新增模型厂商 | 继承 BaseEmbedder、BaseLLMProvider 或 BaseRerankProvider，覆盖 backend、default_model、base_url、api_key_env，必要时覆盖 max_batch_size | RerankSettings 已并入 MemorySettings |
| 新增关系库后端 | 继承 BaseDB，提供 start、_missing_tables 与连接串构造，并在 config_db 的 Literal 与 build_database 放行 | 同步与异步实现需分别登记 |
| 调整打分策略 | `util/scoring.py` 的 score_and_rank 与 get_bm25_params | 权重与阈值常量在 `memory/handler.py` 顶部 |
| 新增转换工具 | `util/numberic_util.py`、`util/datetime_util.py` | 存储层已统一经 str_to_int 与 ensure_datetime 归一 |
| 顶层门面 | `memories.py` 目前为空：应由 loader_config.load_config 装配 embedder、llm、reranker、db、vector，再构造 MemoryHandler | 需先修 P0 的签名与异步继承问题 |
| 对话历史滚动压缩 | 门控与裁剪进 `memory/handler.py`，摘要复用 LLMHandler，阈值进 `core/config_memory.py` | 摘要识别与轮次切分宜做纯函数模块，避免被 splitIntoTurns 计入轮次 |

## 11. 本次扫描结论

- 分层骨架完整：core 配置、能力层三套同构注册表、memory 编排、storage 双方言双同步异步，职责边界清晰；util 包已把指纹、ID 归一与时间转换上收，跨层重复实现减少。
- 主要风险仍是跨层调用签名未对齐：引擎 insert 已改为从 payload 取租户，但 store 与 handler 未同步，写入、读取、召回三条主链在运行时仍会中断。
- 次要风险是 payload 键名三套并存（text、data 以及 text_lemma、text_lemmatized 的两组拼写），更新链路会把向量库的正文与词项列覆盖为空，属静默数据损坏。
- 门面未实现，MemoryHandler 的 get 与 page_query 为空体，异步实现未覆写；对外 API 尚不完整。
- 测试已按新契约重写并改为 test_integration_ 前缀，纯函数与契约用例与集成用例解耦，但本地缺 sqlalchemy，规则化回归仍需在完整 hatch 环境中执行。
