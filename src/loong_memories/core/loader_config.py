# 配置装载：把 JSON 字符串、YAML 文件与环境变量合成一份 MemorySettings。

# 三类来源按优先级从低到高叠加，同名键做深度合并，最终交给 pydantic 校验：
#   各 Settings 的字段默认值（最低）
#   YAML 文件
#   JSON 字符串
#   环境变量（最高）

# 环境变量以 LOONG_MEMORIES__ 为前缀，双下划线分隔层级，例如：
#   LOONG_MEMORIES__DB__BACKEND 映射为 db.backend
#   LOONG_MEMORIES__DB__POOL_CONFIG__POOL_SIZE 映射为 db.pool_config.pool_size

# 环境变量的值默认保持字符串，由 pydantic 完成类型转换（true、5、1.5 均可）；
# 仅当值以 [ 或 { 开头时才按 JSON 解析，用于表达列表与嵌套对象。

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loong_memories.core.config_memory import MemorySettings
from loong_memories.core.exceptions import ConfigError, OptionalDependencyError

logger = logging.getLogger(__name__)

# 环境变量前缀：以双下划线结尾，与嵌套分隔符保持一致。
DEFAULT_ENV_PREFIX = 'LOONG_MEMORIES__'

# 环境变量中嵌套层级的切分符。
ENV_NESTED_DELIMITER = '__'


def load_config(json_text: Optional[str] = None,
                yaml_path: Optional[str] = None,
                use_env: bool = True,
                env_prefix: str = DEFAULT_ENV_PREFIX) -> MemorySettings:
    '''装载配置的总入口。

    json_text 为 JSON 字符串，yaml_path 为 YAML 文件路径，use_env 控制是否读取环境变量。
    三者都可缺省，缺省即不叠加该来源；全部缺省时得到一份纯默认配置。

    解析或校验失败统一抛 ConfigError，便于上层单点捕获。
    '''
    merged: Dict[str, Any] = {}

    if yaml_path:
        _deep_merge(merged, load_config_from_yaml(yaml_path))

    if json_text:
        _deep_merge(merged, load_config_from_json(json_text))

    if use_env:
        _deep_merge(merged, load_config_from_env(env_prefix))

    try:
        return MemorySettings.model_validate(merged)
    except Exception as exc:
        raise ConfigError('配置校验失败：' + str(exc)) from exc


def load_config_from_json(text: str) -> Dict[str, Any]:
    '''解析 JSON 字符串，顶层必须是对象。'''
    if not text or not text.strip():
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError('JSON 配置解析失败：' + str(exc)) from exc

    return _ensure_mapping(data, 'JSON 字符串')


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    '''读取 YAML 文件，空文件返回空字典。'''
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ConfigError('YAML 配置文件不存在：' + str(file_path))

    yaml = _import_yaml()

    try:
        with file_path.open('r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigError('YAML 配置文件读取失败：' + str(exc)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError('YAML 语法错误：' + str(exc)) from exc

    return _ensure_mapping(data, 'YAML 文件 ' + str(file_path))


def load_config_from_env(prefix: str = DEFAULT_ENV_PREFIX) -> Dict[str, Any]:
    '''扫描进程环境变量，把前缀匹配的键转换为嵌套字典。'''
    if not prefix:
        raise ConfigError('环境变量前缀不能为空')

    collected: Dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue

        remainder = key[len(prefix):]
        if not remainder:
            logger.warning('忽略缺少字段名的环境变量：' + key)
            continue

        nested_keys = [part.lower() for part in remainder.split(ENV_NESTED_DELIMITER) if part]
        if not nested_keys:
            logger.warning('忽略无法解析层级的环境变量：' + key)
            continue

        _set_nested(collected, nested_keys, _coerce_env_value(raw), key)

    return collected


# ---------- 内部辅助 ----------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    '''把 override 深度合并进 base：同为字典则递归，否则覆盖。'''
    for key, value in override.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_merge(current, value)
        else:
            base[key] = value
    return base


def _ensure_mapping(data: Any, source: str) -> Dict[str, Any]:
    '''校验顶层为对象，空值归一为空字典。'''
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(source + ' 的顶层必须是对象，实际为 ' + type(data).__name__)

    return dict(data)


def _set_nested(target: Dict[str, Any], keys: List[str], value: Any, env_key: str) -> None:
    '''按层级写入；中间层缺失时自动创建，遇到非字典节点则报冲突。'''
    cursor = target
    for name in keys[:-1]:
        existing = cursor.get(name)
        if existing is None:
            existing = {}
            cursor[name] = existing
        elif not isinstance(existing, dict):
            raise ConfigError('环境变量 ' + env_key + ' 与其他变量在字段 ' + name + ' 上冲突')

        cursor = existing

    cursor[keys[-1]] = value


def _coerce_env_value(raw: str) -> Any:
    '''环境变量值默认保持字符串，交由 pydantic 转类型；以 [ 或 { 开头时按 JSON 解析。'''
    text = raw.strip()
    if text[:1] in ('[', '{'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning('环境变量值形如 JSON 但解析失败，按字符串处理：' + raw)

    return raw


def _import_yaml():
    '''惰性导入 PyYAML；未安装时抛 OptionalDependencyError。'''
    try:
        import yaml
    except ImportError as exc:
        raise OptionalDependencyError('读取 YAML 配置需要安装 PyYAML') from exc

    return yaml

