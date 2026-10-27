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
 作用域构造与校验：全部为纯函数，不做任何外部调用。

 约定：user_id、agent_id、run_id 是唯一的身份来源，metadata 中的同名字段一律剥离。
"""


from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from loong_memories.core.exceptions import DataError

logger = logging.getLogger(__name__)

# 可用于作用域的实体参数。
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})
# 不可经 metadata 设置的身份键：三个实体 ID 加上 actor_id。
IDENTITY_KEYS = ENTITY_PARAMS | {"actor_id"}


def validate_entity_id(value: Optional[Any], name: str) -> Optional[str]:
    """校验并归一实体 ID：非字符串强转、去首尾空白、拒空、拒内部空白。"""
    if value is None:
        return None
    # 调用方常传整数主键，统一在此转字符串，保证各流程作用域口径一致。
    if not isinstance(value, str):
        value = str(value)
    trimmed = value.strip()
    if not trimmed:
        raise DataError(name + "不能为空或纯空白")
    if any(char.isspace() for char in trimmed):
        raise DataError(name + "不能包含空白字符")
    return trimmed


def strip_identity_keys(
    metadata: Optional[Dict[str, Any]],
    existing: Optional[Dict[str, Any]] = None,
    *,
    context: str = "metadata",
) -> Dict[str, Any]:
    """剥离 metadata 中的身份键；值未变则静默，值变化则告警并忽略。"""
    if not metadata:
        return {}
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in IDENTITY_KEYS:
            clean[key] = value
            continue
        if existing is not None and value == existing.get(key):
            continue
        logger.warning(context + "：忽略 metadata[" + key + "]，身份字段不可经 metadata 设置")
    return clean


def build_scope(
    *,
    user_id: Optional[Any] = None,
    agent_id: Optional[Any] = None,
    run_id: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """构造「存储用 payload 模板」与「查询用 filters」；至少需要一个实体 ID。"""
    # 身份范围只由实体参数决定，先剥离调用方 metadata 中的同名键。
    payload: Dict[str, Any] = strip_identity_keys(metadata, None, context="add")
    query: Dict[str, Any] = deepcopy(filters) if filters else {}

    user_id = validate_entity_id(user_id, "user_id")
    agent_id = validate_entity_id(agent_id, "agent_id")
    run_id = validate_entity_id(run_id, "run_id")

    provided = []
    values = {
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
    }
    for name, value in values.items():
        if value:
            payload[name] = value
            query[name] = value
            provided.append(name)

    if not provided:
        raise DataError("user_id、agent_id、run_id 至少提供一个")
    return payload, query


def escape_scope_value(value: Any) -> str:
    """转义会话键的结构分隔符，避免不同组合拼出同一个键。"""
    return str(value).replace("%", "%25").replace("&", "%26").replace("=", "%3D")


def session_scope(filters: Dict[str, Any]) -> str:
    """按实体 ID 生成确定性会话键，供消息表归组。"""
    parts = []
    for name in sorted(ENTITY_PARAMS):
        value = filters.get(name)
        if value:
            parts.append(name + "=" + escape_scope_value(value))
    return "&".join(parts)


def validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """校验检索参数：阈值属于零到一，条数为非负整数（排除布尔）。"""
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise DataError("threshold 必须是数字")
        if threshold < 0 or threshold > 1:
            raise DataError("threshold 必须在 0 到 1 之间")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise DataError("top_k 必须是整数")
        if top_k < 0:
            raise DataError("top_k 不能为负")


def validate_query(query: str) -> str:
    """校验并归一检索语句：必须为非空字符串。"""
    if not isinstance(query, str):
        raise DataError("query 必须是字符串")
    trimmed = query.strip()
    if not trimmed:
        raise DataError("query 不能为空或纯空白")
    return trimmed

