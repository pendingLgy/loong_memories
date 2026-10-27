# 数据库后端注册表：支持注册自定义实现，只负责注册与解析，不做实例化。
#
# 注册方式二选一：继承 BaseDB 并声明 backend（自动注册），或调用 register_db 显式登记。
# 注册值可为类对象，也可为点分路径；实例化在 db_engine.build_database 中进行，本模块不持有任何实例。

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

from loong_memories.registry import Ref, Registry
from loong_memories.storage.message.base import BaseDB


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


def resolve_db(name: str) -> Type[BaseDB]:
    "按后端名取得实现类，路径登记的在此惰性导入。"
    return _REGISTRY.resolve(name)
