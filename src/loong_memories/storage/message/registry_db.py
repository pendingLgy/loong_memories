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

from typing import Dict, List, Any

from loong_memories.registry import Ref, Registry

# 内置后端名称到类路径的映射，按需惰性导入。
BUILTIN_DBS: Dict[str, str] = {
    "sqlite": "loong_memories.storage.message.sqlite_db.SQLiteDB",
    "mysql": "loong_memories.storage.message.mysql_db.MySQLDB",
    "pg": "loong_memories.storage.message.pg_db.PGDB",
    "sqlite_async": "loong_memories.storage.message.sqlite_db.AsyncSQLiteDB",
    "mysql_async": "loong_memories.storage.message.mysql_db.AsyncMySQLDB",
    "pg_async": "loong_memories.storage.message.pg_db.AsyncPGDB",
}

_REGISTRY = Registry("数据库后端", BUILTIN_DBS)


def register_db(name: str, ref: Ref) -> None:
    "登记数据库后端：ref 可为类对象，也可为点分路径字符串。"
    _REGISTRY.register(name, ref)


def register_db_provider(name: str):
    "装饰器形式登记数据库后端。"
    return _REGISTRY.decorator(name)


def is_db_registered(name: str) -> bool:
    "判断数据库后端是否已登记。"
    return name in _REGISTRY


def available_dbs() -> List[str]:
    "返回已登记的后端名，按字典序排列。"
    return _REGISTRY.names()


def resolve_db(name: str) -> Any:
    "按后端名取得实现类，路径登记的在此惰性导入。"
    return _REGISTRY.resolve(name)