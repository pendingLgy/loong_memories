'''loader_config.py 测试包：三类配置来源与内部辅助函数。

覆盖范围：
1. 模块常量：环境变量前缀与嵌套分隔符
2. load_config_from_json：空值、正常解析、顶层非对象、语法错误
3. load_config_from_yaml：文件缺失、空文件、注释文件、正常解析、顶层非对象、语法错误
4. load_config_from_env：前缀匹配、多级嵌套、键名小写化、自定义前缀、空前缀
5. 私有辅助：_deep_merge、_ensure_mapping、_set_nested、_coerce_env_value、_import_yaml
6. load_config 总入口：默认值、单来源、优先级叠加、开关、错误包装

loader_config 依赖 MemorySettings，进而依赖 pydantic-settings；该依赖缺失时整个模块跳过。

本仓库 addopts 为 -m 'integration'，运行前需覆盖：
    python -m pytest tests -o addopts= -q -k loader_config
'''

from __future__ import annotations

import builtins
import json
import os

import pytest

from conftest import CONFIG_JSON_TEXT, CONFIG_MEMORY_SETTINGS


try:
    import pydantic_settings
except Exception as e:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 pydantic_settings', allow_module_level=True)


from loong_memories.core import loader_config
from loong_memories.core.exceptions import ConfigError, OptionalDependencyError
from loong_memories.core.loader_config import (
    DEFAULT_ENV_PREFIX,
    ENV_NESTED_DELIMITER,
    load_config,
    load_config_from_env,
    load_config_from_json,
    load_config_from_yaml,
)


def _json(data):
    '''把字典或列表序列化为 JSON 字符串，避免在源码中手写引号。'''
    return json.dumps(data)


def _write_yaml(directory, text, name='config.yml'):
    '''把文本写入临时 YAML 文件，返回其路径字符串。'''
    target = directory / name
    target.write_text(text, encoding='utf-8')
    return str(target)


@pytest.fixture
def clean_env(monkeypatch):
    '''清空所有带默认前缀的环境变量，隔离宿主环境，并返回 monkeypatch 供用例写入。'''
    for key in list(os.environ):
        if key.startswith(DEFAULT_ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ---------- 模块常量 ----------

class TestModuleConstants:
    '''环境变量前缀与嵌套分隔符的契约。'''

    def test_default_env_prefix(self):
        assert DEFAULT_ENV_PREFIX == 'LOONG_MEMORIES__'

    def test_nested_delimiter(self):
        assert ENV_NESTED_DELIMITER == '__'

    def test_prefix_ends_with_delimiter(self):
        assert DEFAULT_ENV_PREFIX.endswith(ENV_NESTED_DELIMITER)


# ---------- 公共方法：load_config_from_json ----------

class TestLoadConfigFromJson:
    '''JSON 字符串解析。'''

    def test_empty_string_returns_empty_dict(self):
        assert load_config_from_json('') == {}

    def test_whitespace_returns_empty_dict(self):
        text = chr(32) * 3 + chr(10) + chr(9)
        assert load_config_from_json(text) == {}

    def test_parses_flat_object(self):
        assert load_config_from_json(_json({'a': 1})) == {'a': 1}

    def test_parses_nested_object(self):
        payload = {'db': {'mysql': {'port': 3307}}}
        assert load_config_from_json(_json(payload)) == payload

    def test_returns_new_dict_each_call(self):
        first = load_config_from_json(_json({'a': 1}))
        first['a'] = 2
        assert load_config_from_json(_json({'a': 1})) == {'a': 1}

    def test_null_top_level_returns_empty_dict(self):
        assert load_config_from_json('null') == {}

    @pytest.mark.parametrize('text', ['[]', '[1, 2]', 'text', '123', 'true'])
    def test_non_object_top_level_raises(self, text):
        with pytest.raises(ConfigError):
            load_config_from_json(text)

    def test_invalid_json_raises(self):
        with pytest.raises(ConfigError):
            load_config_from_json('{broken json}')

    def test_error_message_mentions_json(self):
        with pytest.raises(ConfigError) as info:
            load_config_from_json('{broken json}')
        assert 'JSON' in str(info.value)

    def test_object_with_double_quotes_parsed(self):
        payload = {'llm': {'provider': 'qwen'}}
        assert load_config_from_json(_json(payload)) == payload


# ---------- 公共方法：load_config_from_yaml ----------

class TestLoadConfigFromYaml:
    '''YAML 文件读取。'''

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config_from_yaml(str(tmp_path / 'missing.yml'))

    def test_missing_file_message_mentions_path(self, tmp_path):
        target = str(tmp_path / 'missing.yml')
        with pytest.raises(ConfigError) as info:
            load_config_from_yaml(target)
        assert 'missing.yml' in str(info.value)

    def test_empty_file_returns_empty_dict(self, tmp_path):
        assert load_config_from_yaml(_write_yaml(tmp_path, '')) == {}

    def test_comment_only_file_returns_empty_dict(self, tmp_path):
        text = '# only a comment' + chr(10)
        assert load_config_from_yaml(_write_yaml(tmp_path, text)) == {}

    def test_parses_nested_mapping(self, tmp_path):
        text = 'db:' + chr(10) + '  backend: mysql' + chr(10) + '  mysql:' + chr(10) + '    host: db.example' + chr(10)
        data = load_config_from_yaml(_write_yaml(tmp_path, text))
        assert data['db']['backend'] == 'mysql'
        assert data['db']['mysql']['host'] == 'db.example'

    def test_parses_utf8_content(self, tmp_path):
        text = 'db:' + chr(10) + '  sqlite:' + chr(10) + '    path: 数据库.db' + chr(10)
        data = load_config_from_yaml(_write_yaml(tmp_path, text))
        assert data['db']['sqlite']['path'] == '数据库.db'

    def test_list_top_level_raises(self, tmp_path):
        text = '- a' + chr(10) + '- b' + chr(10)
        with pytest.raises(ConfigError):
            load_config_from_yaml(_write_yaml(tmp_path, text))

    def test_scalar_top_level_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config_from_yaml(_write_yaml(tmp_path, 'just a string'))

    def test_syntax_error_raises(self, tmp_path):
        text = 'a: [1, 2' + chr(10)
        with pytest.raises(ConfigError):
            load_config_from_yaml(_write_yaml(tmp_path, text))

    def test_error_message_mentions_yaml(self, tmp_path):
        text = 'a: [1, 2' + chr(10)
        with pytest.raises(ConfigError) as info:
            load_config_from_yaml(_write_yaml(tmp_path, text))
        assert 'YAML' in str(info.value)


# ---------- 公共方法：load_config_from_env ----------

class TestLoadConfigFromEnv:
    '''环境变量扫描与嵌套还原。'''

    def test_no_matching_vars_returns_empty_dict(self, clean_env):
        assert load_config_from_env() == {}

    def test_single_level_key(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB', 'x')
        assert load_config_from_env() == {'db': 'x'}

    def test_two_levels(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__BACKEND', 'mysql')
        assert load_config_from_env() == {'db': {'backend': 'mysql'}}

    def test_three_levels(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__POOL_CONFIG__POOL_SIZE', '10')
        assert load_config_from_env() == {'db': {'pool_config': {'pool_size': '10'}}}

    def test_keys_are_lowercased(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'LLM__PROVIDER', 'qwen')
        assert load_config_from_env() == {'llm': {'provider': 'qwen'}}

    def test_sibling_keys_share_parent(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__BACKEND', 'mysql')
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__TABLE_PREFIX', 'lm_')
        assert load_config_from_env() == {'db': {'backend': 'mysql', 'table_prefix': 'lm_'}}

    def test_custom_prefix(self, clean_env):
        clean_env.setenv('CUSTOM__DB__BACKEND', 'pg')
        assert load_config_from_env('CUSTOM__') == {'db': {'backend': 'pg'}}

    def test_unrelated_prefix_ignored(self, clean_env):
        clean_env.setenv('OTHER__DB__BACKEND', 'pg')
        assert load_config_from_env() == {}

    def test_empty_prefix_raises(self):
        with pytest.raises(ConfigError):
            load_config_from_env('')

    def test_json_list_value_parsed(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'ANY__ITEMS', _json([1, 2]))
        assert load_config_from_env() == {'any': {'items': [1, 2]}}

    def test_json_object_value_parsed(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'ANY__OBJ', _json({'a': 1}))
        assert load_config_from_env() == {'any': {'obj': {'a': 1}}}

    def test_plain_value_kept_as_string(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__MYSQL__PORT', '3307')
        assert load_config_from_env() == {'db': {'mysql': {'port': '3307'}}}


# ---------- 私有方法：_deep_merge ----------

class TestDeepMerge:
    '''深度合并：同级字典递归，其余覆盖。'''

    def test_adds_new_keys(self):
        base = {'a': 1}
        assert loader_config._deep_merge(base, {'b': 2}) == {'a': 1, 'b': 2}

    def test_overrides_scalar(self):
        assert loader_config._deep_merge({'a': 1}, {'a': 2}) == {'a': 2}

    def test_merges_nested_dicts(self):
        base = {'db': {'mysql': {'host': 'h1', 'port': 1}}}
        override = {'db': {'mysql': {'port': 2}, 'backend': 'mysql'}}
        merged = loader_config._deep_merge(base, override)
        assert merged == {'db': {'mysql': {'host': 'h1', 'port': 2}, 'backend': 'mysql'}}

    def test_type_mismatch_overwrites(self):
        merged = loader_config._deep_merge({'a': {'b': 1}}, {'a': 'x'})
        assert merged == {'a': 'x'}

    def test_returns_base_object(self):
        base = {'a': 1}
        assert loader_config._deep_merge(base, {'b': 2}) is base

    def test_empty_override_keeps_base(self):
        assert loader_config._deep_merge({'a': 1}, {}) == {'a': 1}

    def test_does_not_mutate_override(self):
        override = {'db': {'port': 2}}
        loader_config._deep_merge({'db': {'host': 'h1'}}, override)
        assert override == {'db': {'port': 2}}


# ---------- 私有方法：_ensure_mapping ----------

class TestEnsureMapping:
    '''顶层结构校验。'''

    def test_none_returns_empty_dict(self):
        assert loader_config._ensure_mapping(None, 'src') == {}

    def test_mapping_is_copied(self):
        original = {'a': 1}
        result = loader_config._ensure_mapping(original, 'src')
        assert result == original
        assert result is not original

    @pytest.mark.parametrize('data', [[], [1], 'text', 1, 1.5, True])
    def test_non_mapping_raises(self, data):
        with pytest.raises(ConfigError):
            loader_config._ensure_mapping(data, 'src')

    def test_error_message_contains_source(self):
        with pytest.raises(ConfigError) as info:
            loader_config._ensure_mapping([], 'JSON 字符串')
        assert 'JSON 字符串' in str(info.value)

    def test_error_message_contains_type_name(self):
        with pytest.raises(ConfigError) as info:
            loader_config._ensure_mapping([], 'src')
        assert 'list' in str(info.value)


# ---------- 私有方法：_set_nested ----------

class TestSetNested:
    '''按层级写入嵌套字典。'''

    def test_single_level(self):
        target = {}
        loader_config._set_nested(target, ['a'], 1, 'KEY')
        assert target == {'a': 1}

    def test_creates_intermediate_levels(self):
        target = {}
        loader_config._set_nested(target, ['a', 'b', 'c'], 1, 'KEY')
        assert target == {'a': {'b': {'c': 1}}}

    def test_reuses_existing_dicts(self):
        target = {'a': {'b': {}}}
        loader_config._set_nested(target, ['a', 'b', 'c'], 1, 'KEY')
        assert target == {'a': {'b': {'c': 1}}}

    def test_conflict_with_scalar_raises(self):
        target = {'a': 'x'}
        with pytest.raises(ConfigError):
            loader_config._set_nested(target, ['a', 'b'], 1, 'KEY')

    def test_conflict_message_contains_env_key(self):
        target = {'a': 'x'}
        with pytest.raises(ConfigError) as info:
            loader_config._set_nested(target, ['a', 'b'], 1, 'MY_KEY')
        assert 'MY_KEY' in str(info.value)

    def test_overwrites_existing_leaf(self):
        target = {'a': 1}
        loader_config._set_nested(target, ['a'], 2, 'KEY')
        assert target == {'a': 2}


# ---------- 私有方法：_coerce_env_value ----------

class TestCoerceEnvValue:
    '''环境变量取值策略：默认保持字符串，仅 JSON 容器字面量才解析。'''

    def test_plain_string_kept(self):
        assert loader_config._coerce_env_value('mysql') == 'mysql'

    def test_numeric_string_kept_as_string(self):
        assert loader_config._coerce_env_value('3306') == '3306'

    def test_boolean_string_kept_as_string(self):
        assert loader_config._coerce_env_value('true') == 'true'

    def test_leading_zero_password_kept(self):
        '''全数字口令不能被转成整数，否则前导零会丢失。'''
        assert loader_config._coerce_env_value('0123') == '0123'

    def test_empty_string_kept(self):
        assert loader_config._coerce_env_value('') == ''

    def test_json_list_parsed(self):
        assert loader_config._coerce_env_value(_json([1, 2, 3])) == [1, 2, 3]

    def test_json_object_parsed(self):
        assert loader_config._coerce_env_value(_json({'a': 1})) == {'a': 1}

    def test_leading_space_before_json_still_parsed(self):
        assert loader_config._coerce_env_value('  ' + _json([1])) == [1]

    def test_broken_json_falls_back_to_string(self):
        broken = '[not json'
        assert loader_config._coerce_env_value(broken) == broken

    def test_broken_json_logs_warning(self, caplog):
        with caplog.at_level('WARNING', logger=loader_config.__name__):
            loader_config._coerce_env_value('{broken')
        assert any('JSON' in record.getMessage() for record in caplog.records)


# ---------- 私有方法：_import_yaml ----------

class TestImportYaml:
    '''PyYAML 惰性导入。'''

    def test_returns_yaml_module(self):
        module = loader_config._import_yaml()
        assert hasattr(module, 'safe_load')

    def test_missing_pyyaml_raises_optional_dependency_error(self, monkeypatch):
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == 'yaml':
                raise ImportError('yaml missing')
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(OptionalDependencyError):
            loader_config._import_yaml()


# ---------- 公共方法：load_config ----------


@pytest.fixture(scope='session')
def config_json_text():
    '''四个分区合并后的 JSON 字符串。'''
    return CONFIG_JSON_TEXT

@pytest.fixture(scope='session')
def config_memory_settings():
    '''MemorySettings 实例；缺失 pydantic-settings 时跳过用例。'''
    if CONFIG_MEMORY_SETTINGS is None:
        pytest.skip('构造 MemorySettings 需要安装 pydantic-settings')
    return CONFIG_MEMORY_SETTINGS.model_copy(deep=True)

class TestLoadConfig:
    '''总入口：多来源叠加、优先级与错误包装。'''

    def test_defaults_only(self, clean_env):
        settings = load_config()
        assert settings.db.backend == 'sqlite'
        assert settings.vector.backend == 'milvus'
        assert settings.embedding.enable is False
        assert settings.llm.enable is False

    def test_json_source_applied(self, clean_env):
        settings = load_config(json_text=_json({'db': {'backend': 'mysql'}}))
        assert settings.db.backend == 'mysql'

    def test_unspecified_fields_keep_model_defaults(self, clean_env):
        settings = load_config(json_text=_json({'db': {'backend': 'mysql'}}))
        assert settings.db.mysql.host == 'localhost'
        assert settings.db.pool_config.pool_size == 5

    def test_yaml_source_applied(self, clean_env, tmp_path):
        path = _write_yaml(tmp_path, 'db:' + chr(10) + '  backend: pg' + chr(10))
        settings = load_config(yaml_path=path)
        assert settings.db.backend == 'pg'

    def test_env_source_applied(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__BACKEND', 'mysql')
        settings = load_config()
        assert settings.db.backend == 'mysql'

    def test_use_env_false_ignores_environment(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__BACKEND', 'mysql')
        settings = load_config(use_env=False)
        assert settings.db.backend == 'sqlite'

    def test_env_overrides_json(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'DB__BACKEND', 'pg')
        settings = load_config(json_text=_json({'db': {'backend': 'mysql'}}))
        assert settings.db.backend == 'pg'

    def test_json_overrides_yaml(self, clean_env, tmp_path):
        path = _write_yaml(tmp_path, 'db:' + chr(10) + '  backend: pg' + chr(10))
        settings = load_config(yaml_path=path, json_text=_json({'db': {'backend': 'mysql'}}))
        assert settings.db.backend == 'mysql'

    def test_yaml_and_json_merge_instead_of_replace(self, clean_env, tmp_path):
        path = _write_yaml(tmp_path, 'db:' + chr(10) + '  mysql:' + chr(10) + '    host: yaml-host' + chr(10))
        settings = load_config(yaml_path=path, json_text=_json({'db': {'mysql': {'port': 3307}}}))
        assert settings.db.mysql.host == 'yaml-host'
        assert settings.db.mysql.port == 3307

    def test_env_values_converted_by_pydantic(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'VECTOR__TOP_K', '20')
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'LLM__ENABLE', 'true')
        settings = load_config()
        assert settings.vector.top_k == 20
        assert settings.llm.enable is True

    def test_custom_env_prefix(self, clean_env):
        clean_env.setenv('MYPREFIX__DB__BACKEND', 'mysql')
        settings = load_config(env_prefix='MYPREFIX__')
        assert settings.db.backend == 'mysql'

    def test_empty_json_text_treated_as_absent(self, clean_env):
        settings = load_config(json_text='')
        assert settings.db.backend == 'sqlite'

    def test_invalid_json_raises_config_error(self, clean_env):
        with pytest.raises(ConfigError):
            load_config(json_text='{broken json}')

    def test_missing_yaml_raises_config_error(self, clean_env, tmp_path):
        with pytest.raises(ConfigError):
            load_config(yaml_path=str(tmp_path / 'missing.yml'))

    def test_validation_failure_raises_config_error(self, clean_env):
        '''temperature 超出 0 到 2 的范围，应由 pydantic 拦下并包装成 ConfigError。'''
        with pytest.raises(ConfigError):
            load_config(json_text=_json({'llm': {'temperature': 5}}))

    def test_unknown_field_raises_config_error(self, clean_env):
        '''各 Settings 声明 extra 为 forbid，未知字段必须报错。'''
        with pytest.raises(ConfigError):
            load_config(json_text=_json({'llm': {'unknown_field': 1}}))

    def test_invalid_env_value_raises_config_error(self, clean_env):
        clean_env.setenv(DEFAULT_ENV_PREFIX + 'VECTOR__TOP_K', 'not-a-number')
        with pytest.raises(ConfigError):
            load_config()


    def test_uses_conftest_full_json(self, clean_env, config_json_text):
        settings = load_config(json_text=config_json_text, use_env=False)
        assert settings.db.backend == 'sqlite'
        assert settings.db.table_prefix == 'lm_'
        assert settings.vector.collection == 'test_collection'
        assert settings.vector.top_k == 8

    def test_matches_conftest_memory_settings(self, clean_env, config_json_text, config_memory_settings):
        '''同一份 JSON 经本模块装载与 conftest 直接校验的结果必须一致。'''
        settings = load_config(json_text=config_json_text, use_env=False)
        assert settings.model_dump() == config_memory_settings.model_dump()
