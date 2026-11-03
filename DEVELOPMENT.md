# 开发手册（DEVELOPMENT）

面向在本仓库内改代码的人：环境怎么搭、分层怎么守、跨层契约怎么对、测试怎么跑、坑在哪里。对外使用说明见 [README.md](README.md)。

## 1. 开发环境

| 项 | 现状 |
| --- | --- |
| 运行环境 | Windows 11 + cmd.exe（代码页 936），CPython 3.13.13 |
| 声明支持 | `requires-python` 为 `>=3.8`；实际代码已用 `dict[str, Any]`、`Optional` 与 PEP 604 联合类型等 3.9+ 语法，日常运行 3.13.13 |
| 构建后端 | hatchling，src-layout，版本取自 `src/loong_memories/__about__.py` |
| 环境管理 | hatch（`installer = pip`），环境按 `tool.hatch.envs` 自动创建 |
| 运行期依赖 | `openai==3.26.0`、`PyYAML` 不低于 6.0.3、`pydantic` 不低于 2.5、`pydantic-settings` 不低于 2.1、`requests` 不低于 2.31 |
| 可选依赖组 | parsers、milvus、sqlalchemy、pg、mysql、dev |

### 环境

依赖与环境一律由 hatch 管理，不要手工 `pip install`，也不要自建 venv。`hatch env show` 可查看当前实际生效的环境矩阵。

| 环境 | features | 用途 |
| --- | --- | --- |
| `default` | parsers、sqlalchemy、dev | 日常开发、单元测试、静态检查 |
| `loong_memories_dev` | parsers、sqlalchemy、pg、mysql、milvus、dev | 集成测试，含 MySQL、PostgreSQL、Milvus 驱动 |
| `types` | mypy 不低于 1.0.0（额外依赖） | 类型检查专用 |

```bash
hatch env create                      # 建 default 环境
hatch env create loong_memories_dev   # 建集成环境
hatch env show                        # 查看环境、features 与脚本
hatch shell                           # 进 default 环境
hatch shell loong_memories_dev        # 进集成环境
hatch env prune                       # 清掉不再使用的环境
```

跨环境执行统一用 `环境名:脚本或命令` 语法，例如 `hatch run loong_memories_dev:pytest tests --integration -q`、`hatch run types:check`。三个环境都定义了 `check`、`format`、`test`、`types` 四个脚本，`types` 环境的 `check` 被覆写为 `mypy --install-types --non-interactive`。

需要临时补装包时也走 hatch，避免污染全局：

```bash
hatch run loong_memories_dev:pip list
hatch run loong_memories_dev:pip install 包名
hatch run loong_memories_dev:pip show ragas          # 查看具体包信息
hatch run loong_memories_dev:pipdeptree -p ragas     # 查看具体包依赖树
```

### 依赖缺失的后果

| 依赖 | 归属 features | 缺失时的后果 |
| --- | --- | --- |
| `sqlalchemy` | sqlalchemy | `storage.message.db_engine` 顶层导入失败，`memory.handler`、`memory.stores` 与全部关系库后端不可导入，处理器相关用例无法收集 |
| `pydantic_settings` | 基础依赖，随包安装 | `core.config_memory` 不可导入，配置与处理器测试整模块 skip |
| `pymilvus` | milvus（仅集成环境） | Milvus 向量库用例整模块 skip |
| `aiosqlite`、`aiomysql`、`asyncpg` | sqlalchemy、mysql、pg | 对应异步用例按类 skip |
| `pytest_asyncio` | dev | 异步用例无法收集 |
| `spacy` 与中文模型 | parsers | BM25 词形还原降级为原文，关键词召回变差 |
| `python-dotenv` | 由 pydantic-settings 传递引入 | `test_integration_ai_memory_handler.py` 的 `load_dotenv` 导入失败，整模块收集中断 |

### 中文输出

Windows 控制台代码页为 936。写文件统一 `encoding=` 指定 utf-8 并保持 CRLF；读文件时若输出出现以 `寮` 开头的乱码，说明字节是 UTF-8 但标准输出被按 cp936 解码，用 `sys.stdout.reconfigure(encoding='cp936')` 再打印即可正常显示，不要据此判定文件损坏。

## 2. 仓库约定

| 位置 | 放什么 | 规则 |
| --- | --- | --- |
| `src/loong_memories` | 全部源码 | src-layout，禁止在仓库根新增同名包 |
| `tests` | 单元与集成用例 | 普通文件用 `test_` 前缀，集成文件统一 `test_integration_` 前缀 |
| `.vcl` | 结构与缺陷文档、方案草稿 | `struct.md` 结构快照、`defect_report.md` 缺陷清单、`plans` 方案草稿 |
| `.vcl_cache` | 一次性脚本与中间产物 | 用完即删，禁止把临时脚本散落在仓库根 |

注意：`.gitignore` 只忽略 `db.sqlite3`，并不忽略 `*.db`，仓库根已存在未跟踪的 `test.db`。SQLite 库文件不要生成在仓库根，测试一律用 pytest 的 `tmp_path`。

## 3. 分层与依赖规则

自上而下：`memory` 依赖 `llm`、`embedding`、`reranks`、`storage`、`core`、`util`、`nlp`；`storage` 与 `llm`、`embedding`、`reranks` 依赖 `core`、`util`、`nlp`、`registry`；`core` 只依赖第三方库与标准库。

| 层次 | 允许依赖 | 禁止 |
| --- | --- | --- |
| `core` | pydantic、pydantic-settings、标准库 | 依赖任何业务层 |
| `llm`、`embedding`、`reranks` | `core`、`registry`、`util` | 依赖 `memory`、`storage` |
| `storage` | `core`、`util`、`nlp`、`registry` | 依赖 `memory`、`llm`、`embedding`、`reranks` |
| `util`、`nlp` | 标准库与三方纯函数库 | 依赖任何业务层 |
| `memory` | 以上全部 | 被外部绕过 handler 直接调方言实现 |

约束细节：

- 注册表模块（`registry_*.py`）只登记与解析，不实例化、不持有状态；实例化只发生在 `llm_handler.py`、`embedder_handler.py`、`rerank_handler.py`，以及顶层门面 `memories.py`（关系库走 `resolve_db`、向量库走 `resolve_vector_store` 后直接构造，不再经 `db_engine.build_database`）。`vector_engine.build_vector_store` 仍是 if 分支直连具体类，新增向量后端必须同时改配置 `Literal` 与该分支。
- 方言实现一律在函数内惰性导入 `pymilvus`、`psycopg`、`pymysql`、`aiosqlite`，模块顶层不导入可选依赖，保证缺驱动时仍可导入包。
- 需要标注上层类型时放进 `if TYPE_CHECKING:` 块并保留 `from __future__ import annotations`，禁止为标注写运行时导入（`llm`、`embedding`、`storage.memories` 曾因此循环导入）。
- 可恢复的失败要抛领域异常（`core.exceptions` 下的 `ConfigError`、`DataError`、`StorageError`、`LLMError`、`EmbeddingError`、`OptionalDependencyError`），不要抛裸 `Exception`，也不要静默吞掉。
- 跨层重复实现已上收，直接复用：时间用 `util.datetime_util`（`now_iso`、`ensure_datetime`、`datetime_to_isoformat`），ID 归一用 `util.numberic_util.str_to_int`、`str_to_float`，文本指纹用 `util.str_util.memory_hash`。

### 3.1 跨层契约（改签名前必读）

`MemoryHandler` 与 `AsyncMemoryHandler` 都继承 `MemoryBase`，六个公开方法必须与基类逐项一致：契约用例 `test_public_signatures_match_memory_base` 用 `inspect.signature` 逐项比对参数名与 kind，改一侧就必须同步另一侧。

| 流程 | 最终签名（同步与异步一致） |
| --- | --- |
| `get` | `(memory_id, tenant_id, user_id=None)` 返回 `GetResult` |
| `page_query` | `(tenant_id, user_id=None, page_num=0, page_size=10, *, filter=None)` 返回 `PageQueryResult` |
| `add` | `(messages, tenant_id, user_id=None, *, agent_id=None, metadata=None)` 返回 `AddResult` |
| `search` | `(query, tenant_id, user_id=None, *, filter=None, top_k=None, threshold=None, rerank=False, explain=False)` 返回 `SearchResult` |
| `update` | `(memory_id, tenant_id, user_id=None, *, text=None, metadata=None)` 返回 `UpdateResult` |
| `delete` | `(memory_id, tenant_id, user_id=None)` 返回 `DeleteResult` |

要点：

- `tenant_id` 位置必填，是硬隔离边界；`user_id` 位置可选，但一旦传入必须一路透传到 store 与 engine。其余参数一律关键字专用。过滤参数名是 `filter`（单数），不是 `filters`。
- 构造签名 `__init__(embedder, llm, db, vector, reranker=None)`，不再接收 `settings`。四个依赖任一为 `None` 立即抛 `ConfigError`；`reranker` 可缺省，`rerank=True` 但未注入时静默走原始顺序。
- 返回模型不对称：`AddResult.results`、`SearchResult.results`、`PageQueryResult.results` 是列表；`GetResult.result`、`UpdateResult.result` 是单条 `MemoryItem` 或 `None`；`DeleteResult.id` 为被删 id。
- `MemoryItem` 时间字段名为 `created_at_tz`、`updated_at_tz`，用旧名 `created_at` 构造会直接抛 `ValidationError`。
- `VectorStoreResult` 是只读视图，没有 `payload` 字段，只有 `id`、`tenant_id`、`user_id`、`text`、`text_lemma`、`created_at_tz`、`updated_at_tz`、`score`、`metadata`。要还原成 dict 必须走 `handler._to_payload()`（内部按 `MEMORY_TABLE_WITHOUT_VECTOR_COLUMNS` 取字段）。
- payload 键名以 `stores.build_payload` 为准：`text`、`text_lemma`、`hash`、`created_at_tz`、`updated_at_tz` 加实体 ID。`CORE_KEYS` 里仍写着历史拼写 `text_lemmatized`，更新链路若按它取值会把正文与词项列写成空值，属静默数据损坏。
- `METADATA_REMOVE_KEYS` 与 `SESSION_PARAMS` 定义在 `storage/memories/base.py`，`memory/scope.py` 直接转引，不要在别处另建一份。
- `VectorEngine.insert(vectors, payloads, ids)` 没有 `tenant_id` 参数，租户只经 payload 表达；`update`、`delete`、`get`、`search`、`keyword_search`、`list` 都带 `tenant_id` 位置参数，过滤参数名同样是 `filter`。
- `MemoryStore.get_memory(memory_id, tenant_id, user_id=None)` 返回 `VectorStoreResult` 或 `None`；`update_memory` 与 `delete_memory` 的第三个位置参数之后分别是 `vector` 与 `previous`，异步侧建议一律用关键字传参。
- `insert_memories(entries, session_scope, messages=None)` 从 entry 顶层读 `tenant_id` 与 `user_id`，历史表这两列都是 NOT NULL，entry 少传即在写审计时抛错。
- `scope.build_new_metadata(agent_id=None, metadata=None, filters=None)` 返回 `(new_metadata, query)`，只处理 `agent_id`；`update_metadata_keys` 另有 `context` 关键字只用于告警文案。`SESSION_PARAMS` 只有 `agent_id`，不传 agent_id 时会话键为空串，消息表会退化成按租户归组。

### 3.2 顶层门面契约（memories.py）

`memories.py` 是唯一装配入口。模块级只剩私有辅助函数，公开面是 `MemoriesBase`、`Memories` 与 `AsyncMemories`；两个子类各自只声明两个类属性，`load` 与 `create` 全部写在基类里。

| 入口 | 类属性 | 产物 | 配置要求 |
| --- | --- | --- | --- |
| `Memories` | `is_async=False`、`handler_cls=MemoryHandler` | `MemoryHandler` | `db.backend` 与 `vector.backend` 必须取同步值 |
| `AsyncMemories` | `is_async=True`、`handler_cls=AsyncMemoryHandler` | `AsyncMemoryHandler` | 必须取 `_async` 值 |

公开方法只有两个，签名固定：

| 方法 | 签名 | 说明 |
| --- | --- | --- |
| `load` | `(json_text=None, yaml_path=None, use_env=True, env_prefix=LOONG_MEMORIES__)` | 装载配置后转 `create` |
| `create` | `(settings: MemorySettings)` | 只吃配置，不接受组件注入 |

约束细节：

- `load` 与 `create` **不接收** `embedder` / `llm` / `reranker` / `db` / `vector`；六个分区全部由配置构造，要换实现就改配置，或在测试里直接构造 handler。
- `REQUIRED_PARTITIONS` 是硬依赖清单（`embedding`、`llm`、`db`、`vector`），任一 `enable=False` 抛 `ConfigError`；`rerank` 与 `nlp` 可选，关闭时 handler 分别收到 `reranker=None` 与 `nlp_setting=None`。
- 三层实现族校验按实现类判定，不看 `backend` 名字：`_db_is_async` 用 `isinstance(db, AsyncDBEngine)`，`_engine_is_async` 用 `isinstance(vector, AsyncVectorEngine)`，`_store_is_async` 看底层实现类的 `search` 是否为协程。任一不符即抛 `ConfigError`，文案带实际类名。
- 向量库实现类由 `_store_class(backend, is_async)` 解析：先把 `backend` 的 `_async` 后缀归一，再经 `resolve_vector_store` 取同步实现类；异步入口在同一模块内找「同步类的子类且 `search` 为协程」的类，找不到就报错提示注入门面之外的实现。
- 连接参数由 `_db_kwargs` 与 `_vector_kwargs` 组装：前者按 `backend.startswith` 覆盖 SQLite、MySQL、PG 三分支并附连接池与表前缀，后者按 `backend.startswith(milvus)` 区分，空字符串字段不下传但 `False` 布尔值保留。
- `load` 与 `create` 只装配不启动：同步侧自行 `handler.start()`，异步侧 `await handler.start()`。
- 包根 `__init__` 刻意不导出本模块，保证 `import loong_memories` 不牵入 sqlalchemy。


## 4. 新增一个实现

### 4.1 存储后端

1. 在 `storage/message` 或 `storage/memories` 下新建模块，继承 `BaseDB` 或 `VectorStoreBase`。
2. 类属性声明 `backend`，`__init_subclass__` 会自动登记；也可用 `register_db`、`register_vector_store` 显式登记。关系库由门面经 `resolve_db` 直接解析，注册表是生效的；向量库仍要同步 `vector_engine.build_vector_store` 的 if 分支。
3. 放行配置：在 `core/config_db.py` 或 `core/config_vector.py` 的 `Literal` 里加入该后端名。
4. 在 `vector_engine.build_vector_store` 与两个注册表 `registry_db`、`registry_vector` 中补分支；连接参数由门面 `memories.py` 的 `_vector_kwargs` 与 `_db_kwargs` 组装后透传，参数集合变动要同步门面。
5. 补测试：契约用例（构造、类属性、缺依赖抛 `OptionalDependencyError`）加集成用例，文件名用 `test_integration_` 前缀。

向量库必须遵守的契约：

- `search` 返回的 `score` 越大越相似。距离度量要先转相似度：COSINE 用 `max(0.0, 1.0 - distance)`，L2 用 `1.0 / (1.0 + distance)`。
- `keyword_search` 不支持的实现返回 `None`，不要抛异常，否则双路召回会整体失败。
- 过滤参数名统一为 `filter`，与 `stores.recall` 一致。
- 写库前用 `METADATA_REMOVE_KEYS` 剔除 metadata 里的核心键，否则 `text`、`tenant_id` 等会重复落进 metadata 并在读回时覆盖正文。
- 只更新向量时不得回写空正文：`text`、`text_lemma`、`updated_at_tz` 必须与向量同批更新，否则记录会以空正文写回并从召回结果中消失。

### 4.2 模型厂商

1. 在 `llm`、`embedding` 或 `reranks` 下新建模块，继承 `OpenAILLMProvider` 或 `OpenAIEmbedder`（OpenAI 兼容服务只覆盖 `backend`、`default_model`、`base_url`、`api_key_env`，必要时覆盖 `max_batch_size`）。
2. 在 `registry_llm.BUILTIN_PROVIDERS`、`registry_embedder.BUILTIN_EMBEDDERS`、`registry_rerank` 的登记点写点分路径，或依赖自动注册。
3. `provider` 字段是自由字符串，无需改 `Literal`；用 `available_llm_providers()`、`available_embedders()` 验证是否登记成功。
4. 重排实现必须在失败时降级：按原顺序返回并给 0.5 中立分，不要抛异常打断整条检索链路。

### 4.3 配置项

- 所有配置必须有归属：新增字段落在 `core/config_*.py` 的 pydantic 模型里，模型统一把 `extra` 设为 forbid，字段名拼错要直接报错而不是被忽略。
- 顶层是六分区：`enable` 加 `embedding`、`llm`、`rerank`、`db`、`vector`、`nlp`；`MemorySettings` 校验 `embedding.dim` 与 `vector.vector_size` 一致，改一侧必须同步另一侧。
- 新增字段要写 `description`，说明取值与影响，尤其是超时、并发、阈值这类运维关心的项。
- 环境变量只允许在 `core/loader_config.py` 读取（前缀 `LOONG_MEMORIES__`，层级用双下划线），业务代码禁止直接读 `os.environ`。
- 阈值、批大小、top_k 这类可调参数必须进配置模型。`memory` 分区目前不暴露检索默认值，`DEFAULT_TOP_K`（20）、`DEFAULT_THRESHOLD`（0.1）、`DEFAULT_CONTEXT_TOP_K`（10）、`DEFAULT_KEYWORD_WEIGHT`（0.5）、`DEFAULT_HISTORY_MESSAGES`（10）集中在 `memory/handler.py` 顶部。

## 5. 测试

### 5.1 用例布局

| 文件 | 目标模块 | 门控 |
| --- | --- | --- |
| `test_loader_config.py`（513 行） | `core/loader_config.py` | 缺 `pydantic_settings` 时整模块 skip |
| `test_embedding.py`（283 行） | `embedding` 包 | 缺 `openai` 时整模块 skip |
| `test_llm.py`（299 行） | `llm` 包 | 同上 |
| `test_sqlite_db.py`（521 行） | `storage/message/sqlite_db.py` | 默认执行，库文件落 `tmp_path` |
| `test_integration_stores.py`（992 行） | `memory/stores.py` | `--integration`，另需 sqlalchemy 与关系库、向量库 |
| `test_integration_vector_engine.py`（605 行） | `storage/memories/vector_engine.py` | `--integration` |
| `test_integration_pg_store.py`（785 行） | `storage/memories/pg_store.py` | `--integration`，需 PG 与 pgvector |
| `test_integration_milvus_store.py`（713 行） | `storage/memories/milvus_store.py` | `--integration`，需 Milvus |
| `test_integration_mysql_db.py`（613 行） | `storage/message/mysql_db.py` | `--integration`，需 MySQL |
| `test_integration_pg_db.py`（603 行） | `storage/message/pg_db.py` | `--integration`，需 PG |
| `test_integration_ai_memory_handler.py`（879 行） | `memory/handler.py` 端到端 | `--integration_ai`，需真实关系库、向量库与 LLM 凭据 |
| `test_integration_ai_memories.py`（947 行） | 门面 `memories.py` 端到端 | `--integration_ai`，需真实关系库、向量库与 LLM 凭据 |

`tests.conftest.py`（245 行）提供：`--integration` 与 `--integration_ai` 两个自定义开关、`unit`、`integration`、`integration_ai` 三个标记注册、src 路径注入，以及 `CONFIG_EMBEDDING`、`CONFIG_LLM`、`CONFIG_RERANK`、`CONFIG_NLP`、`CONFIG_DB`、`CONFIG_VECTOR` 六份配置字典、对应 Settings 对象与 `CONFIG_MEMORY_SETTINGS`。注意 `CONFIG_JSON_DICT` 不含顶层 `enable`，走门面的用例必须自己补 `enable=True`。

### 5.2 门控机制

四层叠加，缺任何一层都会误跑或误判：

1. 依赖层：模块顶部 `try: import ...` 加 `except ImportError`，缺依赖时 `pytest.skip(..., allow_module_level=True)`，并打印 `[SKIP INFO]` 说明缺什么。
2. 参数层：模块级判定 `if --integration not in sys.argv:` 后整模块 skip；AI 端到端文件用 `--integration_ai`。模块级 skip 会把同文件内的 Mock 单元用例一起跳过，所以 `test_integration_ai_memory_handler.py` 里的同步与异步 Mock 套件默认也不执行，必须显式传 `--integration_ai`。
3. 标记层：模块级 `pytestmark = pytest.mark.integration`（AI 文件为 `integration_ai`），个别类再挂 `skipif`，例如缺异步客户端时的 `needs_async_client`。
4. 运行期探测：fixture 连不上真实服务时 `pytest.skip` 而不是 fail，避免把环境问题误判成代码缺陷。

### 5.3 运行命令

`pyproject.toml` 当前的 `addopts` 是 `-s -q -rs`（原先的 `-m integration` 那行已注释），所以默认会收集全部用例，集成用例靠文件内的参数门控自行跳过。一律经 hatch 执行：

```bash
# 默认环境：单元用例与离线用例
hatch run pytest tests -q                                 # 全部用例（集成文件整模块 skip）
hatch run pytest tests -q -rs                             # 附带 skip 原因
hatch run pytest tests -q -m unit                         # 只跑单元标记
hatch run pytest tests -q --cov --cov-report=term-missing # 覆盖率
hatch run pytest tests -q -k memories                      # 门面单元用例，缺依赖时整模块 skip

# 集成环境：需要真实 MySQL、PostgreSQL、Milvus
hatch run loong_memories_dev:pytest tests --integration -q -k mysql
hatch run loong_memories_dev:pytest tests --integration -q -k pg
hatch run loong_memories_dev:pytest tests --integration -q -k milvus
hatch run loong_memories_dev:pytest tests --integration -q tests/test_integration_stores.py -v -s
hatch run loong_memories_dev:pytest tests --integration_ai -q -k ai_memory_handler
hatch run loong_memories_dev:pytest tests --integration_ai -q -k ai_memories
```

`hatch run test` 走 pyproject 里的 `test = pytest`，不额外传参时等于 `pytest` 加默认 `addopts`。因为集成门控在文件内部，用 `-m` 反选已经拦不住它们，要跳过就靠不传 `--integration`／`--integration_ai`。

### 5.4 编写约定

- 命名 `test_方法名_场景`，一个用例只断言一个行为。
- 只测公共接口：私有方法（`_start`、`_missing_tables`、`_to_payload` 等）不单独建用例，其逻辑由公共方法间接覆盖。
- 不为抽象基类单独建套件（不出现 `TestBaseDB`、`TestVectorStoreBase`），但父类提供的公共方法必须在子类实例上验证。
- 公开签名一旦调整，必须同步 `memory/base.py`、两个 handler 与 `inspect.signature` 契约用例，三者任一漏改都会让契约用例失败。
- 外部服务隔离手段：随机 `memory_id`、随机 `tenant_id`、固定表名或集合名加 `tmp_path` 库文件，避免用例互相污染。
- 测试替身注意：`MagicMock` 若没有 `keyword_search` 类属性会破坏 `supports_keyword` 判定；需要返回记忆的地方必须构造真实 `VectorStoreResult`，不要伪造 `payload` 属性。
- fixture 一律 `scope=module`；异步 fixture 用 `pytest_asyncio.fixture(scope=module, loop_scope=module)`。
- 断言已有缺陷时用 `xfail(strict=True)` 锁定，修好后该用例会失败提醒删除标记。

注意：依赖由 hatch 环境提供，环境未创建或未同步时相关用例只会 skip；跑之前先 `hatch env create loong_memories_dev`，不要因为看到 skip 就判定用例通过。

## 6. 质量工具

全部走 hatch，脚本已在 pyproject 定义好，无需手工调用 ruff 或 mypy：

```bash
hatch run check                     # ruff check src tests
hatch run format                    # ruff format src tests
hatch run types                     # mypy src tests

# 只检查运行时源码
hatch run ruff check src
hatch run ruff format src

# 类型检查专用环境：先 --install-types 装 stub 再检查
hatch run types:check
hatch run types:check src/loong_memories
```

## 7. 已知坑与规避

这些问题都真实踩过，改动相关代码时按右列规避：

| 坑 | 表现 | 规避 |
| --- | --- | --- |
| payload 键名两套拼写 | 更新后正文变空、记录从召回结果中消失 | 以 `build_payload` 的 `text`、`text_lemma` 为准，不要按 `CORE_KEYS` 的 `text_lemmatized` 取值 |
| `VectorStoreResult` 没有 `payload` | 访问即 `AttributeError` | 用 `handler._to_payload()` 还原；`_to_item` 里再从 metadata 剔除 `text_lemma` |
| 记忆模型时间字段已改名 | 构造即 `ValidationError` | 用 `created_at_tz`、`updated_at_tz`，不要写 `created_at`、`updated_at` |
| 历史与消息表 `tenant_id` 为 NOT NULL | 写审计记录时完整性报错 | `_persist_facts` 产出的 entry 顶层必须带 `tenant_id` 与 `user_id` |
| `GetResult.result` 是单条而非列表 | 按列表处理会报 `TypeError` | 用例与文档都按单条断言，判空用 `is None` |
| 过滤参数叫 `filter` | 传 `filters` 报未知关键字 | handler、store、engine 全链路统一单数 `filter` |
| `MemoryHandler` 不再接收 `settings` | 沿用旧装配方式会 `TypeError` | 注入 `embedder`、`llm`、`db`、`vector`，`reranker` 可选 |
| 模块级 skip 会连坐 | 同文件的 Mock 单元用例也不执行 | 必须传 `--integration_ai` 才收该文件，不要拿默认运行结果当回归结论 |
| 工厂不查注册表 | 新后端已登记仍选中失败 | 关系库已改由门面经 `resolve_db` 解析；向量库仍需同步 `Literal` 与 `build_vector_store` 的 if 分支 |
| 门面不接受组件注入 | 传 `embedder=` 之类会 `TypeError` | `Memories.create` / `load` 只吃配置；测试要用替身就直接构造 `MemoryHandler` |
| `CONFIG_JSON_DICT` 不含顶层 `enable` | 走门面的用例全被「记忆服务未启用」拦下 | 用例自己补 `enable=True`，或直接给 `MemorySettings` 对象 |
| 同步入口包住异步方言 | 运行期拿到未 `await` 的协程 | 门面装配期三层族校验会拦下；直接构造引擎的代码需自行确认 |
| `.gitignore` 未忽略 `*.db` | 仓库根残留未跟踪的 `test.db` | 用例统一落 `tmp_path`，必要时补忽略规则 |
| 中文经管道输出 | 控制台把中文标题显示成以 `寮` 开头的乱码 | 只是解码口径不一致，用 `sys.stdout.reconfigure(encoding='cp936')` 复看，不要据此改文件编码 |

### 改动自检清单

- 改 handler 公开签名：同步 `memory/base.py`、`MemoryHandler`、`AsyncMemoryHandler`、`inspect.signature` 契约用例、`README.md` 的 API 参考表。
- 改返回模型字段：同步 `memory/models.py`、handler 的结果组装、`README.md` 的返回模型表。
- 改存储层签名：同步 `memory/stores.py`、`storage/memories/vector_engine.py`、方言实现与对应集成用例。
- 改配置字段：同步 `core/config_*.py`、`config_memory` 的联动校验、`tests/conftest.py` 的 `CONFIG_*` 字典、`README.md` 的配置章节。
- 新增文件：补 `__init__.py`、补包级导出（如需要）、补测试并归入正确的门控层。
- 改门面契约：同步 `memories.py` 的 `load` 与 `create` 签名、`REQUIRED_PARTITIONS`、三层族判定函数、`README.md` 装配章节与门面契约用例。
- 改后端枚举：同步 `core.config_db` 或 `core.config_vector` 的 `Literal`、门面的 `_db_kwargs` 与 `_vector_kwargs` 与 `_store_class` 的后缀归一、集成用例的后端矩阵。
