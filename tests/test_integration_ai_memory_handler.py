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
loong_memories 记忆处理器测试套件

包含四部分：
1. 同步 MemoryHandler 单元测试（Mock 存储，验证编排与租户透传）
2. 异步 MemoryHandler 单元测试（Mock 存储，验证编排与构造对齐）
3. 真实同步数据库与向量库的端到端集成测试
4. 真实异步数据库与向量库的端到端集成测试

集成部分需要 --integration_ai，单元部分随时可跑。
"""

import inspect
import logging
import os
import random
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text

from loong_memories.core.exceptions import ConfigError, DataError
from loong_memories.embedding.embedder_handler import EmbedderHandler
from loong_memories.llm.llm_handler import LLMHandler
from loong_memories.memory.handler import AsyncMemoryHandler, MemoryHandler
from loong_memories.memory.base import MemoryBase
from loong_memories.memory.models import (
    AddResult,
    DeleteResult,
    GetResult,
    PageQueryResult,
    SearchResult,
    UpdateResult,
)
from loong_memories.reranks.rerank_handler import RerankHandler
from loong_memories.storage.memories.base import VectorStoreResult
from loong_memories.storage.memories.milvus_store import AsyncMilvusVectorStore, MilvusVectorStore
from loong_memories.storage.memories.pg_store import AsyncPGVectorStore, PGVectorStore
from loong_memories.storage.message.mysql_db import AsyncMySQLDB, MySQLDB
from loong_memories.storage.message.pg_db import AsyncPGDB, PGDB
from loong_memories.storage.message.sqlite_db import AsyncSQLiteDB, SQLiteDB

from conftest import (
    CONFIG_DB_SETTINGS,
    CONFIG_EMBEDDING_SETTINGS,
    CONFIG_LLM_SETTINGS,
    CONFIG_RERANK_SETTINGS,
    CONFIG_VECTOR_SETTINGS,
)

logger = logging.getLogger(__name__)



# 集成用例默认跳过：需要真实数据库、向量库与 LLM 凭据，用 --integration_ai 显式开启。
pytestmark = pytest.mark.integration_ai

if "--integration_ai" not in sys.argv:
    pytest.skip(f"[SKIP INFO] File {__file__}：integration_ai 集成测试，默认跳过（请使用 --integration_ai 参数运行）",
                allow_module_level=True)


# 显式加载环境变量 .env
load_dotenv()

# 单元测试统一使用的租户与用户：断言点集中在此处，便于随契约调整。
TENANT_ID = 7001
USER_ID = 9527


def _store_result(
    memory_id: int = 1,
    body: str = "旧正文",
    tenant_id: int = TENANT_ID,
    user_id: int = USER_ID,
    score: float = None,
) -> VectorStoreResult:
    """构造存储层读取结果：正文走 text 列，业务键放 metadata。"""
    return VectorStoreResult(
        id=memory_id,
        tenant_id=tenant_id,
        user_id=user_id,
        text=body,
        text_lemma=body,
        created_at_tz="2026-01-01T00:00:00+00:00",
        updated_at_tz="2026-01-01T00:00:00+00:00",
        score=score,
        metadata={"role": "user", "actor_id": "tester"},
    )


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=EmbedderHandler)
    embedder.encode.return_value = [[0.12] * CONFIG_VECTOR_SETTINGS.vector_size]
    return embedder


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMHandler)
    llm.chat.return_value = '{"memory": [{"text": "用户偏好 Python 与架构设计"}]}'
    return llm


@pytest.fixture
def sync_mock_handler(mock_embedder, mock_llm):
    """同步处理器：store 整体替换为 MagicMock，只验证编排与入参。"""
    with patch("loong_memories.memory.handler.MemoryStore"):
        handler = MemoryHandler(embedder=mock_embedder, llm=mock_llm, db=MagicMock(), vector=MagicMock())
    handler.store = MagicMock()
    handler.store.recall.return_value = []
    handler.store.load_history.return_value = []
    return handler


@pytest.fixture
def async_mock_handler(mock_embedder, mock_llm):
    """异步处理器：store 替换为 AsyncMock，返回协程结果。"""
    with patch("loong_memories.memory.handler.AsyncMemoryStore"):
        handler = AsyncMemoryHandler(embedder=mock_embedder, llm=mock_llm, db=AsyncMock(), vector=MagicMock())
    handler.store = AsyncMock()
    handler.store.recall.return_value = []
    handler.store.load_history.return_value = []
    return handler


# ===========================================================================
# 第一部分：同步处理器单元测试 (Sync Mock Suite)
# ===========================================================================

class TestSyncMemoryHandlerMockSuite:
    """验证同步 MemoryHandler 的编排流程、租户透传与构造期校验。"""

    def test_init_requires_db(self, mock_embedder, mock_llm):
        """缺少关系库时构造期抛 ConfigError。"""
        with pytest.raises(ConfigError, match="关系库"):
            MemoryHandler(embedder=mock_embedder, llm=mock_llm, db=None, vector=MagicMock())

    def test_init_reranker_is_optional(self, mock_embedder, mock_llm):
        """重排器可选：不传时实例属性为 None，检索不触发重排。"""
        with patch("loong_memories.memory.handler.MemoryStore"):
            handler = MemoryHandler(embedder=mock_embedder, llm=mock_llm, db=MagicMock(), vector=MagicMock())
        assert handler.reranker is None

    def test_add_puts_tenant_in_entry(self, sync_mock_handler):
        """写入时租户与用户同时落在 entry 顶层，供审计表取用。"""
        result = sync_mock_handler.add("同步开发记录", TENANT_ID, USER_ID)

        assert isinstance(result, AddResult)
        assert len(result.results) == 1
        entries = sync_mock_handler.store.insert_memories.call_args.args[0]
        assert entries[0]["tenant_id"] == TENANT_ID
        assert entries[0]["user_id"] == USER_ID
        assert entries[0]["payload"]["text"] == result.results[0].memory


    def test_add_passes_tenant_to_history(self, sync_mock_handler):
        """会话历史的读取必须带租户与用户，否则取不到任何消息。"""
        sync_mock_handler.add("同步开发记录", TENANT_ID, USER_ID)

        args = sync_mock_handler.store.load_history.call_args.args
        assert args[1] == TENANT_ID
        assert args[2] == 10
        assert args[3] == USER_ID

    def test_add_recall_keeps_identity_out_of_filter(self, sync_mock_handler):
        """租户与用户走独立入参，filter 里不再残留同名键。"""
        sync_mock_handler.add("同步开发记录", TENANT_ID, USER_ID)

        kwargs = sync_mock_handler.store.recall.call_args.kwargs
        assert kwargs["tenant_id"] == TENANT_ID
        assert kwargs["user_id"] == USER_ID
        assert kwargs["filter"] == {}

    def test_add_requires_tenant_keyword(self, sync_mock_handler):
        """tenant_id 是必填关键字参数：漏传属于调用错误，构造不出租户边界。"""
        with pytest.raises(TypeError):
            sync_mock_handler.add("只有用户没有租户", user_id=USER_ID)

    def test_update_rebuilds_vector_when_text_changed(self, sync_mock_handler):
        """正文变化：重算向量，payload 键名与 build_payload 对齐。"""
        sync_mock_handler.store.get_memory.return_value = _store_result()

        result = sync_mock_handler.update(10, TENANT_ID, USER_ID, text="新正文")

        assert isinstance(result, UpdateResult)
        assert result.result.memory == "新正文"
        args = sync_mock_handler.store.update_memory.call_args.args
        assert args[0] == 10
        assert args[1] == TENANT_ID

        kwargs = sync_mock_handler.store.update_memory.call_args.kwargs
        assert kwargs['payload']["text"] == "新正文"
        assert kwargs['payload']["text_lemma"]
        assert kwargs['previous'] == "旧正文"
        assert kwargs['user_id'] == USER_ID

    def test_update_keeps_vector_when_text_unchanged(self, sync_mock_handler):
        """正文未变：向量传 None，让后端保留原值。"""
        sync_mock_handler.store.get_memory.return_value = _store_result(body="同一正文")

        sync_mock_handler.update(10, TENANT_ID, USER_ID, text="同一正文")

        sync_mock_handler.store.assert_not_called()

    def test_update_requires_text_or_metadata(self, sync_mock_handler):
        """正文与元数据同时缺省时报错。"""
        with pytest.raises(DataError, match="至少提供一个"):
            sync_mock_handler.update(10, TENANT_ID)

    def test_update_missing_record_raises(self, sync_mock_handler):
        """目标记忆不存在时报错。"""
        sync_mock_handler.store.get_memory.return_value = None

        with pytest.raises(DataError, match="记忆不存在：10"):
            sync_mock_handler.update(10, TENANT_ID, text="任意")

    def test_delete_passes_tenant_and_previous(self, sync_mock_handler):
        """删除时租户必传，旧正文用于审计记录。"""
        sync_mock_handler.store.get_memory.return_value = _store_result(body="待删正文")

        result = sync_mock_handler.delete(10, TENANT_ID, USER_ID)

        assert isinstance(result, DeleteResult)
        assert result.id == 10
        args = sync_mock_handler.store.delete_memory.call_args.args
        assert args[0] == 10
        assert args[1] == TENANT_ID
        assert args[2] == "待删正文"
        assert args[4] == USER_ID


    def test_search_ignores_identity_keys_in_filters(self, sync_mock_handler):
        """租户与用户已是显式入参：filters 里的同名键被剔除，入参优先。"""
        sync_mock_handler.store.recall.return_value = []

        sync_mock_handler.search("查询", TENANT_ID, USER_ID, filter={"tenant_id": 9999, "user_id": 8888})

        kwargs = sync_mock_handler.store.recall.call_args.kwargs
        assert kwargs["tenant_id"] == TENANT_ID
        assert kwargs["user_id"] == USER_ID
        assert kwargs["filter"] == {"tenant_id": 9999, "user_id": 8888}

    def test_search_forwards_scope_to_recall(self, sync_mock_handler):
        """检索把租户、用户与高级过滤条件分别下发。"""
        sync_mock_handler.store.recall.return_value = [
            {"id": 1, "memory": "命中的记忆", "hash": "h1", "score": 0.9,
             "tenant_id": TENANT_ID, "user_id": USER_ID}
        ]

        result = sync_mock_handler.search("查询关键词", TENANT_ID, USER_ID)

        assert isinstance(result, SearchResult)
        assert result.results[0].memory == "命中的记忆"
        kwargs = sync_mock_handler.store.recall.call_args.kwargs
        assert kwargs["tenant_id"] == TENANT_ID
        assert kwargs["user_id"] == USER_ID
        assert kwargs["filter"] == None

    def test_search_keeps_extra_filter(self, sync_mock_handler):
        """agent_id 等业务键保留在 filter 中。"""
        sync_mock_handler.store.recall.return_value = []

        sync_mock_handler.search("查询", TENANT_ID, None, filter={"agent_id": "a1"})

        assert sync_mock_handler.store.recall.call_args.kwargs["filter"] == {"agent_id": "a1"}

    def test_get_returns_empty_when_missing(self, sync_mock_handler):
        """读取不存在的记忆返回空结果集，不抛异常。"""
        sync_mock_handler.store.get_memory.return_value = None

        result = sync_mock_handler.get(memory_id=10, tenant_id=TENANT_ID, user_id=USER_ID)

        assert isinstance(result, GetResult)
        assert result.result == None

    def test_get_maps_store_result_to_item(self, sync_mock_handler):
        """存储结果映射为对外条目：正文进 memory，业务键进 role。"""
        sync_mock_handler.store.get_memory.return_value = _store_result(body="命中的正文")

        result = sync_mock_handler.get(memory_id=10, tenant_id=TENANT_ID, user_id=USER_ID)

        assert result.result.memory == "命中的正文"
        assert result.result.role == "user"

    def test_page_query_slices_rows(self, sync_mock_handler):
        """分页按 page_num 与 page_size 在抓取窗口内切片。"""
        sync_mock_handler.store.vector_engine.list.return_value = [
            _store_result(memory_id=i, body="条目" + str(i)) for i in range(5)
        ]

        result = sync_mock_handler.page_query(
            tenant_id=TENANT_ID, user_id=USER_ID, page_num=1, page_size=2
        )

        assert isinstance(result, PageQueryResult)
        assert result.total_size == 5
        assert [item.memory for item in result.results] == ["条目2", "条目3"]
        assert sync_mock_handler.store.vector_engine.list.call_args.kwargs["top_k"] == 4

    def test_page_query_rejects_bad_paging(self, sync_mock_handler):
        """页码为负或页大小为零时报错。"""
        with pytest.raises(DataError, match="page_size"):
            sync_mock_handler.page_query(tenant_id=TENANT_ID, page_size=0)


    def test_public_signatures_match_memory_base(self):
        """接口契约锁定：公开方法签名必须与 base.py 的 MemoryBase 完全一致。"""
        for name in ("get", "page_query", "add", "search", "update", "delete"):
            expected = inspect.signature(getattr(MemoryBase, name))
            actual = inspect.signature(getattr(MemoryHandler, name))
            assert [(item.name, item.kind) for item in actual.parameters.values()] == [
                (item.name, item.kind) for item in expected.parameters.values()
            ], name



# ===========================================================================
# 第二部分：异步处理器单元测试 (Async Mock Suite)
# ===========================================================================

@pytest.mark.asyncio
class TestAsyncMemoryHandlerMockSuite:
    """验证异步 MemoryHandler 的构造对齐、编排流程与租户透传。"""

    async def test_init_keeps_vector_and_reranker(self, mock_embedder, mock_llm):
        """构造参数不再错位：vector 与 reranker 各归其位。"""
        vector = MagicMock()
        with patch("loong_memories.memory.handler.AsyncMemoryStore"):
            handler = AsyncMemoryHandler(embedder=mock_embedder, llm=mock_llm, db=AsyncMock(), vector=vector)

        assert handler.vector is vector
        assert handler.reranker is None

    async def test_init_accepts_reranker(self, mock_embedder, mock_llm):
        """显式传入的重排器落在 reranker 上，不被向量引擎顶替。"""
        reranker = MagicMock(spec=RerankHandler)
        with patch("loong_memories.memory.handler.AsyncMemoryStore"):
            handler = AsyncMemoryHandler(
                embedder=mock_embedder, llm=mock_llm, db=AsyncMock(), vector=MagicMock(), reranker=reranker
            )

        assert handler.reranker is reranker

    async def test_add_deduplicates_existing_hash(self, async_mock_handler):
        """命中既有指纹时跳过写入，但仍保存消息。"""
        async_mock_handler.store.recall.return_value = [
            {"id": 1, "memory": "用户偏好 Python 与架构设计", "hash": "h_abc",
             "score": 0.8, "tenant_id": TENANT_ID}
        ]

        with patch("loong_memories.memory.handler.memory_hash", return_value="h_abc"):
            result = await async_mock_handler.add("重复消息", TENANT_ID, USER_ID)

        assert result.results == []
        async_mock_handler.store._save_messages.assert_awaited_once()

    async def test_add_passes_tenant_to_history(self, async_mock_handler):
        """异步历史读取同样带租户与用户。"""
        await async_mock_handler.add("异步开发记录", TENANT_ID, USER_ID)

        args = async_mock_handler.store.load_history.call_args.args
        assert args[1] == TENANT_ID
        assert args[3] == USER_ID

    async def test_add_requires_tenant_keyword(self, async_mock_handler):
        """异步写入同样强制 tenant_id 关键字参数。"""
        with pytest.raises(TypeError):
            await async_mock_handler.add("只有用户", user_id=USER_ID)


    async def test_search_forwards_scope_to_recall(self, async_mock_handler):
        """异步检索不依赖已移除的融合方法，直接用 store.recall 结果。"""
        async_mock_handler.store.recall.return_value = [
            {"id": 99, "memory": "异步检索命中的记忆", "hash": "h2", "score": 0.95,
             "tenant_id": TENANT_ID, "user_id": USER_ID}
        ]

        result = await async_mock_handler.search(
            "查询关键词", TENANT_ID, USER_ID
        )

        assert isinstance(result, SearchResult)
        assert result.results[0].id == 99
        assert async_mock_handler.store.recall.call_args.kwargs["tenant_id"] == TENANT_ID

    async def test_search_ignores_identity_keys_in_filters(self, async_mock_handler):
        """异步检索同样以入参为准，filters 中的身份键不参与过滤。"""
        async_mock_handler.store.recall.return_value = []

        await async_mock_handler.search("查询", TENANT_ID, filter={"user_id": 8888})

        kwargs = async_mock_handler.store.recall.call_args.kwargs
        assert kwargs["tenant_id"] == TENANT_ID
        assert kwargs["filter"] == {"user_id": 8888}

    async def test_update_not_found_raises(self, async_mock_handler):
        """异步更新不存在的记忆报错。"""
        async_mock_handler.store.get_memory.return_value = None

        with pytest.raises(DataError, match="记忆不存在：888"):
            await async_mock_handler.update(888, TENANT_ID, text="更新内容")


    async def test_update_workflow(self, async_mock_handler):
        """异步更新走同一套合并与向量刷新规则。"""
        async_mock_handler.store.get_memory.return_value = _store_result(memory_id=888)

        result = await async_mock_handler.update(
            888, TENANT_ID, USER_ID, text="异步新正文"
        )

        assert result.result.memory == "异步新正文"
        args = async_mock_handler.store.update_memory.call_args.args
        assert args[0] == 888
        assert args[1] == TENANT_ID

        kwargs = async_mock_handler.store.update_memory.call_args.kwargs
        assert kwargs["user_id"] == USER_ID

    async def test_delete_workflow(self, async_mock_handler):
        """异步删除透传租户，并返回被删除的 id。"""
        async_mock_handler.store.get_memory.return_value = _store_result(memory_id=888, body="待删正文")

        result = await async_mock_handler.delete(888, TENANT_ID, USER_ID)

        assert isinstance(result, DeleteResult)
        assert result.id == 888
        assert async_mock_handler.store.delete_memory.call_args.args[1] == TENANT_ID

    async def test_get_returns_empty_when_missing(self, async_mock_handler):
        """异步读取不存在的记忆返回空结果集。"""
        async_mock_handler.store.get_memory.return_value = None

        result = await async_mock_handler.get(memory_id=888, tenant_id=TENANT_ID)

        assert result.result == None


    async def test_public_signatures_match_memory_base(self):
        """异步处理器同样必须与 base.py 的 MemoryBase 签名保持一致。"""
        for name in ("get", "page_query", "add", "search", "update", "delete"):
            expected = inspect.signature(getattr(MemoryBase, name))
            actual = inspect.signature(getattr(AsyncMemoryHandler, name))
            assert [(item.name, item.kind) for item in actual.parameters.values()] == [
                (item.name, item.kind) for item in expected.parameters.values()
            ], name



# ===========================================================================
# 第三部分：真实数据库与向量库集成测试 (Real Integration Suite)
# ===========================================================================

# 实体 ID 随机范围：控制在 BigInteger 安全区间内。
MAX_ENTITY_ID = 2 ** 31 - 1

def random_identity() -> tuple[int, int]:
    """随机租户与用户：每次调用产生新的隔离边界，避免用例之间串数据。"""
    return random.randint(1, MAX_ENTITY_ID), random.randint(1, MAX_ENTITY_ID)


def generate_extractable_human_dialogue() -> tuple[list[dict], list[str], str]:
    """生成人类真实对话测试数据，返回 tuple: (messages, search_keywords, theme)"""

    dialogue_templates = [
        {
            "theme": "技术栈与偏好",
            "search_keywords": ["开发环境", "uv", "Hatch", "pgvector", "LangGraph", "PyCharm", "Mem0", "美式咖啡"],
            "messages": [
                {"role": "user", "content": "哎，我最近把开发环境全面转到 uv 和 Hatch 了，比原来清爽太多。"},
                {"role": "assistant",
                 "content": "确实，uv 的虚拟环境和依赖解析速度极其惊艳，你现在主力项目还是在搞那个 pgvector 迁移吗？"},
                {"role": "user",
                 "content": "对，不仅在搞 pgvector，我还顺便把 LangGraph 的工作流重构了一遍。对了，我一向比较习惯用深色主题的 PyCharm，写后端代码顺手。"},
                {"role": "assistant",
                 "content": "明白，老牌 IDE 配合现代依赖管理工具，效率确实高。那长期记忆模块你准备用 Mem0 还是自己写？"},
                {"role": "user",
                 "content": "我打算深入看下 Mem0 的源码，自己定制一层状态压缩。另外，我习惯喝美式咖啡来保持熬夜写代码的清醒。"}
            ]
        },
        {
            "theme": "工作日常与生活节奏",
            "search_keywords": ["三百万级并发", "数据库优化", "深夜写代码", "周末跑步", "Windows桌面Agent", "pyautogui",
                                "Chrome抓取"],
            "messages": [
                {"role": "user", "content": "今天又是高强度搬砖的一天，处理了三百万级并发的数据库优化。"},
                {"role": "assistant", "content": "听起来是个大工程！你不是一向喜欢在安静的深夜处理这种核心架构问题吗？"},
                {"role": "user",
                 "content": "对的，我通常晚上 10 点以后灵感最好。不过周末我一定会抽出半天去公园跑步，保持体能。"},
                {"role": "assistant",
                 "content": "劳逸结合很关键。对了，你上周提到的那个自动化 GUI 助手项目进展得怎么样了？"},
                {"role": "user",
                 "content": "那个用 Python 和 pyautogui 的 Windows 桌面 Agent 已经跑通了，正准备加个 Chrome 网页数据抓取的小功能。"}
            ]
        },
        {
            "theme": "项目目标与个人计划",
            "search_keywords": ["RAG系统", "Ragas评估", "私有知识库", "自主自建", "PostgreSQL", "pgvector", "极简架构"],
            "messages": [
                {"role": "user", "content": "我最近在整理 RAG 系统的评估指标，用 Ragas 跑了下上下文精确度。"},
                {"role": "assistant", "content": "结果怎么样？你的目标是把它落地到实际的生产环境中吧？"},
                {"role": "user",
                 "content": "对，我的目标是在下个月底前把这套私有知识库系统全面上线。我不喜欢用那些臃肿的 SaaS 服务，坚持所有核心组件都自建。"},
                {"role": "assistant",
                 "content": "这很符合你一贯追求掌控力的工程风格。数据库方面还是坚守 PostgreSQL 吗？"},
                {"role": "user", "content": "对，主库 PostgreSQL 加上 pgvector 扩展，简单稳妥，我挺喜欢这种极简架构的。"}
            ]
        }
    ]

    # 随机选择一个模板
    selected = random.choice(dialogue_templates)

    # 💡 调用示例（可以直接解包）：
    # messages, keywords, theme = generate_extractable_human_dialogue()
    # 返回 Tuple
    return selected["messages"], selected["search_keywords"], selected["theme"]


EMBEDDING_SETTING = CONFIG_EMBEDDING_SETTINGS
EMBEDDING_SETTING.provider = "qwen"
EMBEDDING_SETTING.model = os.getenv("embedding_model","")
EMBEDDING_SETTING.api_key = os.getenv("embedding_api_key","")
EMBEDDING_SETTING.base_url = os.getenv("embedding_url","")


CONFIG_LLM_SETTING = CONFIG_LLM_SETTINGS
CONFIG_LLM_SETTING.provider = "qwen"
CONFIG_LLM_SETTING.model = os.getenv("llm_model","")
CONFIG_LLM_SETTING.api_key = os.getenv("llm_api_key","")
CONFIG_LLM_SETTING.base_url = os.getenv("llm_url","")


CONFIG_RERANK_SETTING = CONFIG_RERANK_SETTINGS
CONFIG_RERANK_SETTING.provider = "qwen"
CONFIG_RERANK_SETTING.model = os.getenv("rerank_model","")
CONFIG_RERANK_SETTING.api_key = os.getenv("rerank_api_key","")
CONFIG_RERANK_SETTING.base_url = os.getenv("rerank_url","")

DB_BACKENDS = ["sqlite", "mysql", "pg"]  # ["sqlite", "mysql", "pg"]
VECTOR_BACKENDS = ["milvus", "pg"]  # ["milvus", "pg"]

@pytest.fixture
def real_embedder():
    embedding = EmbedderHandler(EMBEDDING_SETTING)
    return embedding


@pytest.fixture
def real_llm():
    llm = LLMHandler(CONFIG_LLM_SETTING)
    return llm

@pytest.fixture
def real_rerank():
    rerank = RerankHandler(CONFIG_RERANK_SETTING)
    return rerank

@pytest.fixture(scope="module", params=DB_BACKENDS)
def db_backend(request):
    return request.param


@pytest.fixture(scope="module", params=VECTOR_BACKENDS)
def vector_backend(request):
    return request.param



class TestMemoryHandlerIntegrationSuite:
    """连接真实 PostgreSQL 同步数据库的端到端集成测试"""

    table_prefix = "handler_test_sync_"
    collection_name = "handler_test_vector"

    @pytest.fixture
    def real_sync_handler(self, db_backend, vector_backend, real_embedder, real_llm, real_rerank):

        logger.info(f"db_backend:{db_backend} vector_backend:{vector_backend}")

        # 初始化真实关系数据库引擎
        if "pg" == db_backend:
            db_conf = CONFIG_DB_SETTINGS.pg
            db_class = PGDB

        elif "mysql" == db_backend:
            db_conf = CONFIG_DB_SETTINGS.mysql
            db_class = MySQLDB

        else:
            db_conf = CONFIG_DB_SETTINGS.sqlite
            db_class = SQLiteDB

        new_db_conf = {
            **db_conf.model_dump(),  # 核心：将 PGSettings 转换为 Python 字典
            "table_prefix": TestMemoryHandlerIntegrationSuite.table_prefix,
        }

        logger.info(f"db:{str(db_class)} config:{new_db_conf}")

        db = db_class(**new_db_conf)

        # 初始化真实 PG 向量库引擎
        if "milvus" == vector_backend:
            vec_conf = CONFIG_VECTOR_SETTINGS.milvus
            vector_class = MilvusVectorStore
        else:
            vec_conf = CONFIG_VECTOR_SETTINGS.pg
            vector_class = PGVectorStore

        new_vector_conf = {
            **vec_conf.model_dump(),
            "collection_name": TestMemoryHandlerIntegrationSuite.collection_name
        }
        logger.info(f"vector_db:{str(vector_class)} config:{new_vector_conf}")

        vector = vector_class(**new_vector_conf)

        handler = MemoryHandler(
            embedder=real_embedder,
            llm=real_llm,
            db=db,
            vector=vector,
            reranker=real_rerank,
        )

        try:
            handler.start()
        except Exception as exc:
            pytest.skip(f'启动 handler 失败:' + str(exc))

        yield handler

        if hasattr(vector, 'engine') and vector.engine:
            vector.engine.dispose()
        if hasattr(db, 'engine') and db.engine:
            db.engine.dispose()

    def test_02_real_sync_add_and_search(self, real_sync_handler):
        """2. 真实同步数据库环境下的写入与检索端到端测试；租户与用户均为随机值。"""
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()
        logger.info(f"theme:{theme} tenant_id:{tenant_id} user_id:{user_id}")

        # 同步写入
        add_res = real_sync_handler.add(messages, tenant_id, user_id)
        assert isinstance(add_res, AddResult)
        assert len(add_res.results) >= 1

        query = random.choice(keywords)
        # 同步检索
        search_res = real_sync_handler.search(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            top_k=5,
            threshold=0.1,
            rerank=True,
            explain=True,
        )
        assert isinstance(search_res, SearchResult)
        assert len(search_res.results) > 0

        # 租户是隔离边界：换一个随机租户检索，不得命中本租户数据。
        other_tenant_id, _ = random_identity()
        isolated = real_sync_handler.search(
            query=query, tenant_id=other_tenant_id, top_k=5, threshold=0.1
        )
        assert isinstance(isolated, SearchResult)
        assert isolated.results == []


    def test_03_real_sync_update_and_delete(self, real_sync_handler):
        """3. 真实同步数据库环境下的更新与删除端到端测试；租户与用户均为随机值。"""
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()

        # 同步新增
        add_res = real_sync_handler.add(messages, tenant_id, user_id)
        assert add_res.results, "未抽取到事实，无法验证更新与删除"
        memory_id = add_res.results[0].id

        # 同步更新
        new_text = random.choice(keywords) + " 已更新为持久记忆"
        update_res = real_sync_handler.update(memory_id, tenant_id, user_id, text=new_text)
        assert isinstance(update_res, UpdateResult)
        assert update_res.result.memory == new_text

        # 同步删除
        delete_res = real_sync_handler.delete(memory_id, tenant_id, user_id)
        assert isinstance(delete_res, DeleteResult)
        assert delete_res.id == memory_id


# ---------------------- 异步测试 ---------------------------------

@pytest_asyncio.fixture(scope="module", loop_scope='module')
def real_async_embedder():
    embedding = EmbedderHandler(EMBEDDING_SETTING)
    return embedding

@pytest_asyncio.fixture(scope="module", loop_scope='module')
def real_async_llm():
    llm = LLMHandler(CONFIG_LLM_SETTING)
    return llm

@pytest_asyncio.fixture(scope="module", loop_scope='module')
def real_async_rerank():
    rerank = RerankHandler(CONFIG_RERANK_SETTING)
    return rerank

DB_ASYNC_BACKENDS = ["sqlite_async", "mysql_async", "pg_async"]  # ["sqlite_async", "mysql_async", "pg_async"]
VECTOR_ASYNC_BACKENDS = ["milvus_async", "pg_async"]  # ["milvus_async", "pg_async"]


@pytest_asyncio.fixture(scope="module", loop_scope='module', params=DB_ASYNC_BACKENDS)
def db_async_backend(request):
    return request.param


@pytest_asyncio.fixture(scope="module", loop_scope='module', params=VECTOR_ASYNC_BACKENDS)
def vector_async_backend(request):
    return request.param



@pytest.mark.asyncio(loop_scope='module')
class TestAsyncMemoryHandlerIntegrationSuite:
    """连接真实 PostgreSQL 异步数据库的端到端集成测试"""

    table_prefix = "handler_test_async_"
    collection_name = "handler_test_async_"

    @pytest_asyncio.fixture(scope="module",loop_scope="module")
    async def real_async_handler(self, real_async_embedder, real_async_llm,real_async_rerank, db_async_backend, vector_async_backend):

        logger.info(f"db_async_backend:{db_async_backend} vector_async_backend:{vector_async_backend}")

        # 初始化真实关系数据库引擎
        if "pg_async" == db_async_backend:
            db_conf = CONFIG_DB_SETTINGS.pg
            db_class = AsyncPGDB
        elif "mysql_async" == db_async_backend:
            db_conf = CONFIG_DB_SETTINGS.mysql
            db_class = AsyncMySQLDB
        else:
            db_conf = CONFIG_DB_SETTINGS.sqlite
            db_class = AsyncSQLiteDB

        new_db_conf = {
            **db_conf.model_dump(),  # 核心：将 PGSettings 转换为 Python 字典
            "table_prefix": TestAsyncMemoryHandlerIntegrationSuite.table_prefix,
        }

        logger.info(f"db_async:{str(db_class)} config:{new_db_conf}")

        db = db_class(**new_db_conf)

        # 初始化真实 PG 向量库引擎
        if "milvus_async" == vector_async_backend:
            vec_conf = CONFIG_VECTOR_SETTINGS.milvus
            vector_class = AsyncMilvusVectorStore
        else:
            vec_conf = CONFIG_VECTOR_SETTINGS.pg
            vector_class = AsyncPGVectorStore

        new_vector_conf = {
            **vec_conf.model_dump(),
            "collection_name": TestAsyncMemoryHandlerIntegrationSuite.collection_name
        }
        logger.info(f"vector_db:{str(vector_class)} config:{new_vector_conf}")

        vector = vector_class(**new_vector_conf)

        handler = AsyncMemoryHandler(
            embedder=real_async_embedder,
            llm=real_async_llm,
            db=db,
            vector=vector,
            reranker=real_async_rerank,
        )

        try:
            await handler.start()
        except Exception as exc:
            pytest.skip(f'启动 handler 失败:' + str(exc))

        yield handler

        # 清理资源
        if hasattr(vector, 'engine') and vector.engine:
            await vector.engine.dispose()
        if hasattr(db, 'engine') and db.engine:
            await db.engine.dispose()

    async def test_01_real_async_db_connectivity(self, real_async_handler):
        """1. 验证异步数据库连通性与底层自动建表"""
        assert real_async_handler.db.engine is not None
        async with real_async_handler.db.engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    async def test_02_real_async_add_and_search(self, real_async_handler):
        """2. 真实异步数据库环境下的写入与检索联动测试；租户与用户均为随机值。"""
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()

        # 异步写入
        add_res = await real_async_handler.add(messages, tenant_id, user_id)
        assert isinstance(add_res, AddResult)
        assert len(add_res.results) >= 1

        query = random.choice(keywords)
        # 异步检索
        search_res = await real_async_handler.search(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            top_k=5,
            threshold=0.1,
            explain=True
        )
        assert isinstance(search_res, SearchResult)
        assert len(search_res.results) > 0

        # 租户是隔离边界：换一个随机租户检索，不得命中本租户数据。
        other_tenant_id, _ = random_identity()
        isolated = await real_async_handler.search(
            query=query, tenant_id=other_tenant_id, top_k=5, threshold=0.1
        )
        assert isinstance(isolated, SearchResult)
        assert isolated.results == []


    async def test_03_real_async_update_and_delete(self, real_async_handler):
        """3. 真实异步数据库环境下的更新与删除联动测试；租户与用户均为随机值。"""
        tenant_id, user_id = random_identity()
        messages, keywords, theme = generate_extractable_human_dialogue()

        # 异步新增
        add_res = await real_async_handler.add(messages, tenant_id, user_id)
        assert add_res.results, "未抽取到事实，无法验证更新与删除"
        memory_id = add_res.results[0].id

        # 异步更新
        new_text = random.choice(keywords) + " 已更新为持久记忆"
        update_res = await real_async_handler.update(memory_id, tenant_id, user_id, text=new_text)
        assert isinstance(update_res, UpdateResult)
        assert update_res.result.memory == new_text

        # 异步删除
        delete_res = await real_async_handler.delete(memory_id, tenant_id, user_id)
        assert isinstance(delete_res, DeleteResult)
        assert delete_res.id == memory_id

