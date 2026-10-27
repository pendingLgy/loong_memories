# 领域异常基类，统一上层捕获入口。

from __future__ import annotations


class BaseError(Exception):
    """所有领域异常的基类。"""


class ConfigError(BaseError):
    """配置缺失或非法。"""

class DataError(BaseError):
    """数据校验失败。"""


class OptionalDependencyError(BaseError):
    """可选依赖未安装。"""


class EmbeddingError(BaseError):
    """向量化失败。"""

class LLMError(BaseError):
    """大模型调用失败。"""


class StorageError(BaseError):
    """存储读写失败。"""
