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
    领域异常基类，统一上层捕获入口。
"""

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
