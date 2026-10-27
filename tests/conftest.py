'''测试公共配置：自定义 pytest 选项、标记注册，以及按模块拆分的 JSON 配置与对应 Settings 对象。

pyproject.toml 的 addopts 为 -m 'integration'，默认只收集 integration 标记的用例。
运行单元测试请显式覆盖 addopts：

    python -m pytest tests -o addopts='' -q

JSON 配置按 MemorySettings 的分区拆分：embedding、llm、db、vector 各自一份。
每份 JSON 都转换为对应的 Settings 对象，供用例直接断言字段值与约束校验结果。
'''

from __future__ import annotations

import json
import pathlib
import sys


# src-layout 未安装时，把 src 加入导入路径，保证 loong_memories 可导入。
_SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from loong_memories.core.config_db import DBSettings
from loong_memories.core.config_llm import EmbeddingSettings, LLMSettings
from loong_memories.core.config_vector import VectorSettings


def pytest_addoption(parser):
    parser.addoption(
        '--integration',
        action='store_true',
        default=False,
        help='运行真实调用外部服务的集成测试',
    )
    parser.addoption(
        '--eval',
        action='store_true',
        default=False,
        help='评分用例：真实调用 Milvus 与 PG 完成写入与回查',
    )


def pytest_configure(config):
    '''注册本仓库使用的自定义标记，避免 PytestUnknownMarkWarning。'''
    config.addinivalue_line('markers', 'unit: 不依赖外部服务的单元测试')
    config.addinivalue_line('markers', 'integration: 需要真实外部服务的集成测试')


# ---------- JSON 配置：公共工具 ----------

def _dumps(data):
    '''把配置字典序列化为 JSON 字符串，保留中文与可读缩进。'''
    return json.dumps(data, ensure_ascii=False, indent=2)

# ---------- JSON 配置：embedding 分区 ----------

# 对应 EmbeddingSettings；extra='forbid'，字段名必须与模型严格一致。
CONFIG_EMBEDDING = {
    'enable': True,
    'provider': 'openai',
    'model': 'text-embedding-3-small',
    'batch_size': 64,
    'dim': 1024,
    'api_key': 'sk-test-embedding',
    'base_url': 'https://api.example.com/v1',
    'timeout': 30.0,
}

CONFIG_EMBEDDING_JSON = _dumps(CONFIG_EMBEDDING)


# ---------- JSON 配置：llm 分区 ----------

# 对应 LLMSettings；temperature 取值需在 0 到 2 之间。
CONFIG_LLM = {
    'enable': True,
    'provider': 'deepseek',
    'model': 'deepseek-chat',
    'temperature': 0.3,
    'max_tokens': 2048,
    'api_key': 'sk-test-llm',
    'base_url': 'https://api.example.com/v1',
    'timeout': 60.0,
}

CONFIG_LLM_JSON = _dumps(CONFIG_LLM)


# ---------- JSON 配置：db 分区 ----------

# 对应 DBSettings；backend 受 Literal 约束，可选 sqlite、mysql、pg 及带 _async 后缀的取值。
CONFIG_DB = {
    'enable': True,
    'backend': 'sqlite',
    'table_prefix': 'lm_',
    'sqlite': {
        'path': 'F:\\workspace\\code\\python\\loong_memories\\test.db',
        # ---------------- SQLite 可选参数 ----------------
        'timeout': 30.0,
        'isolation_level': '',
        'check_same_thread': False,
    },
    'mysql': {
        'host': '127.0.0.1',
        'port': 3306,
        'user': 'root',
        'password': '123456',
        'database': 'autotest_rag_data',
        'charset': 'utf8mb4',
    },
    'pg': {
        'host': '127.0.0.1',
        'port': 5432,
        'user': 'root',
        'password': '123456',
        'database': 'autotest_rag_data',
        'connection_string': '',
        # ---------------- PostgreSQL 可选连接参数 ----------------
        'sslmode': 'disable',
        'connect_timeout': 5,
        'application_name': 'loong_memories_service',
        'options': '',
        'target_session_attrs': 'any',

    },
    'pool_config': {
        'print_sql': False,
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 5,
        'pool_recycle': 1800,
        'pool_pre_ping': True,
    },
}

CONFIG_DB_JSON = _dumps(CONFIG_DB)

# ---------- JSON 配置：vector 分区 ----------

# 对应 VectorSettings；metric_type 受 Literal 约束，可选 COSINE、L2、IP。
CONFIG_VECTOR = {
    'enable': True,
    'backend': 'milvus',
    'collection': 'test_collection',
    'vector_size': 1024,
    'top_k': 8,
    'metric_type': 'COSINE',
    'milvus': {
        'uri': 'http://localhost:19530',
        'token': '',
        'db_name': 'autotest_rag_data',
    },
    'pg': {
        'connection_string': '',
        'host': '127.0.0.1',
        'port': 5432,
        'user': 'root',
        'password': '123456',
        'database': 'autotest_rag_data',
        'use_hnsw': True,
        'use_diskann': False,
        'sslmode': 'disable',
    },
    'pool_config': {
        'print_sql': False,
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 5,
        'pool_recycle': 1800,
        'pool_pre_ping': False,
    },
}

CONFIG_VECTOR_JSON = _dumps(CONFIG_VECTOR)


# ---------- JSON 配置：合并后的完整配置 ----------

# 四个分区合并，作为 MemorySettings 的完整输入；各分区此处为浅引用，
# 用例需要改动时请取下方对应的 fixture（返回深拷贝）。
CONFIG_JSON_DICT = {
    'embedding': CONFIG_EMBEDDING,
    'llm': CONFIG_LLM,
    'db': CONFIG_DB,
    'vector': CONFIG_VECTOR,
}

CONFIG_JSON_TEXT = _dumps(CONFIG_JSON_DICT)

# ---------- Settings 对象：由 JSON 直接反序列化 ----------

# 各分区一次性完成反序列化与校验，字段非法时在导入阶段即抛 ValidationError，
# 便于尽早发现测试数据与模型定义不一致。
CONFIG_EMBEDDING_SETTINGS = EmbeddingSettings.model_validate_json(CONFIG_EMBEDDING_JSON)
CONFIG_LLM_SETTINGS = LLMSettings.model_validate_json(CONFIG_LLM_JSON)
CONFIG_DB_SETTINGS = DBSettings.model_validate_json(CONFIG_DB_JSON)
CONFIG_VECTOR_SETTINGS = VectorSettings.model_validate_json(CONFIG_VECTOR_JSON)


def _load_memory_settings():
    '''合并配置构造 MemorySettings；缺失 pydantic-settings 时返回 None。'''
    try:
        from loong_memories.core.config_memory import MemorySettings
    except ImportError:
        return None

    return MemorySettings.model_validate(CONFIG_JSON_DICT)


# 顶层聚合配置；依赖 pydantic-settings，未安装时为 None，用例应直接引用该常量。
CONFIG_MEMORY_SETTINGS = _load_memory_settings()
