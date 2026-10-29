# 开发手册（DEVELOPMENT）

面向在本仓库内改代码的人：环境怎么搭、分层怎么守、测试怎么跑、坑在哪里。对外使用说明见 [README.md](README.md)。

## 1. 开发环境

| 项 | 现状 |
| --- | --- |
| 运行环境 | Windows 11 + cmd.exe（代码页 936），CPython 3.13.13 |
| 声明支持 | `requires-python` 为 `>=3.8`（pyproject，实际运行 3.13.13） |
| 构建后端 | hatchling，src-layout，版本取自 `src/loong_memories/__about__.py` |
| 环境管理 | hatch（`installer = pip`），环境按 `tool.hatch.envs` 自动创建 |

### 环境

依赖与环境一律由 hatch 管理，不要手工 `pip install`，也不要自建 venv。`hatch env show` 可查看当前实际生效的环境矩阵。

| 环境 | features | 用途 |
| --- | --- | --- |
| `default` | dev、parsers、sqlalchemy | 日常开发、单元测试、静态检查 |
| `loong_memories_dev` | dev、milvus、mysql、parsers、pg、sqlalchemy | 集成测试，含 MySQL / PostgreSQL / Milvus 驱动 |
| `types` | dev、parsers、sqlalchemy，外加 `mypy>=1.0.0` | 类型检查专用 |

三个环境都暴露 `check` / `format` / `test` / `types` 四个脚本（后两个环境继承 `default` 的脚本定义），`types` 把 `check` 覆写为 `mypy --install-types --non-interactive`。

```bash
hatch env create                      # 建 default 环境
hatch env create loong_memories_dev   # 建集成环境
hatch env show                        # 查看环境、features 与脚本
hatch shell                           # 进 default 环境
hatch shell loong_memories_dev        # 进集成环境
hatch env prune                       # 清掉不再使用的环境
```

跨环境执行统一用 `<环境名>:<脚本或命令>` 语法，例如 `hatch run loong_memories_dev:pytest tests --integration -q`、`hatch run types:check`。

需要临时补装包时也走 hatch，避免污染全局：

```bash
hatch run loong_memories_dev:pip list
hatch run loong_memories_dev:pip install <package>

hatch run loong_memories_dev:pip install pipdeptree # 安装依赖树查看库
hatch run loong_memories_dev:pipdeptree  # 查看依赖树

hatch run loong_memories_dev:pip show ragas # 查看具体包信息
hatch run loong_memories_dev:pipdeptree -p ragas  # 查看具体包依赖树
```

| 依赖 | 归属 features | 缺失时的后果 |
| --- | --- | --- |
| `sqlalchemy` | `sqlalchemy` | `storage.message.db_engine` 顶层导入失败，记忆层与全部关系库后端不可用 |
| `pydantic_settings` | 基础依赖，随包安装 | `core.config_memory` 不可导入，配置相关测试整模块 skip |
| `pymilvus` | `milvus`（仅 `loong_memories_dev`） | Milvus 向量库用例整模块 skip |
| `aiosqlite` / `aiomysql` / `asyncpg` | `sqlalchemy` / `mysql` / `pg` | 对应的异步用例按类 skip |
| `pytest_asyncio` | `dev` | 异步用例无法收集 |
| `spacy` 与中文模型 | `parsers` | BM25 词形还原降级为原文，关键词检索召回变差 |

### 中文输出

Windows 控制台代码页为 936，读中文时设置 `PYTHONIOENCODING=gbk`，写文件时统一 `encoding='utf-8'` 并使用 `open(..., encoding='utf-8')` 或 `pathlib.Path.write_text(encoding='utf-8')`，不要依赖默认编码。

## 2. 仓库约定

| 位置 | 放什么 | 规则 |
| --- | --- | --- |
| `src/loong_memories` | 全部源码 | src-layout，禁止在仓库根新增同名包 |
| `tests` | 单元与集成用例 | 与源码模块一一对应，文件名 `test_` 加模块名 |
| `.vcl` | 结构文档与计划 | `struct.md` 为结构快照，`plans` 为方案草稿 |
| `.vcl_cache` | 一次性脚本与中间产物 | 用完即删，禁止把临时脚本散落在仓库根 |

注意：`.gitignore` 未忽略 `*.db`，SQLite 库文件不要生成在仓库根，测试一律用 pytest 的 `tmp_path`。

## 3. 分层与依赖规则

自上而下：`memory` 依赖 `llm`、`embedding`、`storage`、`core`；`storage` 与 `llm`、`embedding` 依赖 `core`；`core` 只依赖第三方库。

| 层次 | 允许依赖 | 禁止 |
| --- | --- | --- |
| `core` | pydantic、标准库 | 依赖任何业务层 |
| `llm` / `embedding` | `core`、`registry` | 依赖 `memory`、`storage` |
| `storage` | `core`、`util`、`nlp`、`registry` | 依赖 `memory`、`llm`、`embedding` |
| `memory` | 以上全部 | 直接被外部绕过 handler 调用方言实现 |

约束细节：

- 注册表模块（`registry_*.py`）只登记与解析，不实例化、不持有状态；实例化在 `*_handler.py`、`vector_engine.py`、`db_engine.build_database` 中完成。
- 方言实现一律在函数内惰性导入，模块顶层不导入 `pymilvus`、`psycopg`、`pymysql`、`aiosqlite` 之类的可选依赖，保证缺依赖时仍可导入包。
- 需要类型标注引用上层类型时，放进 `if TYPE_CHECKING:` 块并保留 `from __future__ import annotations`，禁止为标注写运行时导入（`llm`、`embedding`、`storage.memories` 曾因此出现循环导入，已按此方案修复）。
- 可恢复的失败要抛领域异常（`core.exceptions` 下的 `ConfigError` / `DataError` / `StorageError` / `LLMError` / `EmbeddingError` / `OptionalDependencyError`），不要抛裸 `Exception`，也不要静默吞掉。

## 4. 新增一个实现

### 4.1 存储后端

1. 在 `storage/message` 或 `storage/memories` 下新建模块，继承 `BaseDB` 或 `VectorStoreBase`。
2. 类属性声明 `backend = "your_backend"`，`__init_subclass__` 会自动登记；也可以用 `register_db` / `register_vector_store` 显式登记。
3. 放行配置：在 `core/config_db.py` 或 `core/config_vector.py` 的 `Literal` 里加入该后端名。
4. 在 `db_engine.build_database` 或相关工厂里补分支，透传连接池参数。
5. 补测试：契约用例（构造、类属性、缺依赖抛 `OptionalDependencyError`）加集成用例。

向量库必须遵守契约：`search` 返回的 `score` 越大越相似。距离度量要先转相似度（COSINE 用 `max(0.0, 1.0 - distance)`，L2 用 `1.0 / (1.0 + distance)`），`keyword_search` 不支持的实现返回 `None` 而不是抛异常。

### 4.2 模型厂商

1. 在 `llm` 或 `embedding` 下新建模块，继承 `OpenAILLMProvider` 或 `OpenAIEmbedder`（OpenAI 兼容服务只覆盖 `backend`、`default_model`、`base_url`、`api_key_env`、`max_batch_size`）。
2. 在 `registry_llm.BUILTIN_PROVIDERS` / `registry_embedder.BUILTIN_EMBEDDERS` 登记点分路径，或依赖自动注册。
3. `provider` 字段是自由字符串，无需改 `Literal`；用 `available_llm_providers()`、`available_embedders()` 验证是否登记成功。

### 4.3 配置项

- 所有配置必须有归属：新增字段落在 `core/config_*.py` 的 pydantic 模型里，模型统一 `extra = "forbid"`，字段名拼错要直接报错而不是被忽略。
- 新增字段要写 `description` 说明取值与影响，尤其是超时、并发、阈值这类运维关心的项。
- 环境变量只允许在 `core/loader_config.py` 读取，业务代码禁止直接 `os.environ`。
- 阈值、批大小、top_k 这类可调参数必须进配置模型，不要散落在业务逻辑里硬编码。

## 5. 测试

### 5.1 用例布局

| 文件 | 目标模块 | 门控 |
| --- | --- | --- |
| `test_loader_config.py` | `core/loader_config.py` | 缺 `pydantic_settings` 时整模块 skip |
| `test_sqlite_db.py` | `storage/message/sqlite_db.py` | 默认执行，库文件落 `tmp_path` |
| `test_mysql_db.py` | `storage/message/mysql_db.py` | 整文件 integration，另需 MySQL 与驱动 |
| `test_pg_db.py` | `storage/message/pg_db.py` | 整文件 integration，异步用例另需 `asyncpg` |
| `test_embedding.py` | `embedding` 包 | 缺 `openai` 时整模块 skip |
| `test_llm.py` | `llm` 包 | 同上 |
| `test_milvus_store.py` | `storage/memories/milvus_store.py` | 整文件 integration，异步用例另需 `AsyncMilvusClient` |
| `test_pg_store.py` | `storage/memories/pg_store.py` | 整文件 integration |

`tests/conftest.py` 提供：`--integration` 与 `--eval` 两个自定义命令行开关、`unit` / `integration` 标记注册、src 路径注入，以及 `CONFIG_EMBEDDING` / `CONFIG_LLM` / `CONFIG_DB` / `CONFIG_VECTOR` 四份配置字典和对应的 Settings 对象。

### 5.2 门控机制

三层叠加，缺任何一层都会误跑：

1. 依赖层：模块顶部 `try: import ...` 加 `except ImportError: pytest.skip(..., allow_module_level=True)`，缺依赖整文件 skip 并打印 `[SKIP INFO]`。
2. 标记层：集成模块写 `pytestmark = pytest.mark.integration`，个别类再挂 `skipif`（如 `needs_asyncpg`、`needs_async_client`）。
3. 参数层：`if '--integration' not in sys.argv: pytest.skip(..., allow_module_level=True)`。

### 5.3 运行命令

`pyproject.toml` 当前的 `addopts` 是 `-s -q -rs -m 'integration'`，即默认只收集 integration 用例。跑单元用例必须覆盖 `addopts`，并且一律经 hatch：

```bash
# 默认环境：单元用例与离线用例
hatch run pytest tests -o addopts= -q                 # 全部用例
hatch run pytest tests -o addopts= -q -m unit         # 只跑单元用例
hatch run pytest tests -o addopts= -q -rs -m unit     # 附带 skip 原因
hatch run test                                        # 等价 pytest，受 addopts 约束，只跑 integration

# 覆盖率
hatch run pytest tests -o addopts= -q --cov --cov-report=term-missing

# 集成环境：需要真实 MySQL / PostgreSQL / Milvus
hatch run loong_memories_dev:pytest tests --integration -q
hatch run loong_memories_dev:pytest tests --integration -q -k mysql
hatch run loong_memories_dev:pytest tests --integration -q -k pg
hatch run loong_memories_dev:pytest tests --integration -q -k milvus #只运行测试函数名称中包含 milvus 的测试用例

hatch run loong_memories_dev:pytest -m 'integration' --integration tests/test_milvus_store.py -v -s
hatch run loong_memories_dev:pytest -m 'integration' --integration tests/test_integration_stores.py::TestAsyncMemoryStoreIntegration -v -s
hatch run loong_memories_dev:pytest -m 'integration' --integration tests/test_integration_stores.py::TestAsyncMemoryStoreIntegration::test_real_store_insert_and_get -v -s
hatch run loong_memories_dev:pytest -m 'not integration' tests/test_milvus_store.py -v -s

```

`hatch run` 的 `-o addopts=` 覆盖对 hatch 脚本同样生效；`hatch run test` 走的是 pyproject 里的 `test = "pytest"`，不额外传参时等于 `pytest` 加默认 `addopts`。

### 5.4 编写约定

- 命名 `test_{方法名}_{场景}`，一个用例只断言一个行为。
- 只测公共接口：私有方法（`_start`、`_missing_tables`、`_parse` 等）不单独建用例，其逻辑由公共方法间接覆盖。
- 不为抽象基类单独建测试套件（不出现 `TestBaseDB`、`TestVectorStoreBase`），但父类提供的公共方法必须在子类实例上验证。
- 外部服务隔离手段：随机 `memory_id`、随机 `tenant_id`、固定表名或集合名加 `tmp_path` 库文件，避免用例互相污染。
- fixture 一律 `scope=module`；异步 fixture 用 `pytest_asyncio.fixture(scope=module, loop_scope=module)`。
- 断言已有缺陷时用 `pytest.mark.xfail(strict=True)` 锁定，修好后该用例会失败提醒删除标记。

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


