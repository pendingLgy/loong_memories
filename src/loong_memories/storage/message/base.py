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

from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional

from sqlalchemy import Engine

from loong_memories.storage.message.schema import HistoryRecord, Message


class BaseDB(ABC):
    """数据库统一接口：声明 backend 后自动登记，配置改后端名即可切换。"""

    # 配置中的数据库后端名；子类置为该名字即自动登记。
    backend: ClassVar[Optional[str]] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """子类声明 backend 后自动登记，无需额外注册调用。"""
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("backend")
        if isinstance(name, str) and name:
            # 局部导入 register_db，彻底避免顶层循环导入
            from loong_memories.storage.message.registry_db import register_db
            register_db(name, cls)

    @abstractmethod
    def _start(self) -> Engine:
        """启动数据库连接"""

    @abstractmethod
    def add_history(self, record: HistoryRecord) -> HistoryRecord:
        """追加一条历史记录，返回补全 id 后的对象。"""

    @abstractmethod
    def batch_add_history(self, records: List[HistoryRecord]) -> List[HistoryRecord]:
        """批量追加历史记录，返回补全 id 后的对象列表。"""

    @abstractmethod
    def get_history(self, memory_id: int) -> List[HistoryRecord]:
        """按 memory_id 取历史，按时间升序。"""

    # ---------------- 会话消息 ----------------

    @abstractmethod
    def save_messages(self, messages: List[Message], keep_last: int = 10) -> List[Message]:
        """写入消息，并只保留各会话最近的 keep_last 条。"""

    @abstractmethod
    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Message]:
        """按时间升序取会话最近 limit 条消息。"""