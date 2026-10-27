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
 数据库表结构：用 SQLAlchemy Core 定义，方言差异由引擎自动适配。

 时间列以 ISO 字符串存储，字典序即时间序，省去时区与类型转换。
"""


from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import BigInteger, Column, Index, MetaData, SmallInteger, String, Table, Text, TIMESTAMP

HISTORY_TABLE_SUFFIX = "history"
MESSAGES_TABLE_SUFFIX = "messages"


def get_history_table(table_metadata: MetaData, prefix: str = "") -> Any:
    history_table_name = prefix + HISTORY_TABLE_SUFFIX

    HISTORY_TABLE = Table(
        history_table_name,
        table_metadata,
        Column("id", BigInteger, primary_key=True),
        Column("memory_id", BigInteger, nullable=True),
        Column("old_memory", Text, nullable=True),
        Column("new_memory", Text, nullable=True),
        Column("event", String(32), nullable=True),
        Column("created_at_tz", TIMESTAMP(timezone=True), nullable=True),
        Column("updated_at_tz", TIMESTAMP(timezone=True), nullable=True),
        Column("is_deleted", SmallInteger, nullable=False, default=0),
        Column("actor_id", String(64), nullable=True),
        Column("role", String(32), nullable=True),
        Index(prefix + "ix_history_memory_id", "memory_id"),
    )

    return HISTORY_TABLE


def get_messages_table(table_metadata: MetaData, prefix: str = "") -> Any:
    messages_table_name = prefix + MESSAGES_TABLE_SUFFIX

    MESSAGES_TABLE = Table(
        messages_table_name,
        table_metadata,
        Column("id", BigInteger, primary_key=True),
        Column("session_scope", String(191), nullable=True),
        Column("role", String(32), nullable=True),
        Column("content", Text, nullable=True),
        Column("name", String(191), nullable=True),
        Column("created_at_tz", TIMESTAMP(timezone=True), nullable=True),
        Index(prefix + "ix_messages_scope_created", "session_scope", "created_at_tz"),
    )

    return MESSAGES_TABLE


# ---------------- 数据对象 ----------------

class HistoryRecord(BaseModel):
    """一条历史记录：变更前后内容、事件类型与操作者。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    memory_id: Optional[int] = None
    old_memory: Optional[str] = None
    new_memory: Optional[str] = None
    event: Optional[str] = None
    created_at_tz: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at_tz: Optional[datetime] = None
    is_deleted: bool = False
    actor_id: Optional[str] = None
    role: Optional[str] = None


class Message(BaseModel):
    """一条会话消息：角色、内容与所属会话。"""

    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    session_scope: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    created_at_tz: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
