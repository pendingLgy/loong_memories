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

from __future__ import annotations

import importlib
from typing import Any, List, Mapping, Type, Union

from loong_memories.core.exceptions import ConfigError

# 注册值：点分路径字符串或类本身。
Ref = Union[str, type]


class Registry(dict):
    """名字到实现的映射，支持惰性导入与统一实例化。"""

    def __init__(self, kind: str, builtins: Mapping[str, Ref]) -> None:
        super().__init__(builtins)
        # kind 用于拼装报错信息，如「存储后端」「嵌入模型」。
        self.kind = kind

    def register(self, name: str, ref: Ref) -> None:
        """登记实现：ref 可为类对象，也可为点分路径字符串（首次使用时导入）。"""
        self[name] = ref

    def decorator(self, name: str):
        """装饰器形式登记实现。"""

        def wrapper(cls: type) -> type:
            self[name] = cls
            return cls

        return wrapper

    def names(self) -> List[str]:
        """已登记的名字，按字典序排列。"""
        return sorted(self)

    def resolve(self, name: str) -> Type[Any]:
        """按名字取得实现类，路径登记的在此时惰性导入。"""
        ref = self.get(name)
        if ref is None:
            raise ConfigError(
                "未注册的" + self.kind + "：" + str(name) + "；可用" + self.kind + "：" + ", ".join(self.names())
            )
        if isinstance(ref, str):
            module_path, _, attr = ref.rpartition(".")
            ref = getattr(importlib.import_module(module_path), attr)
        return ref
