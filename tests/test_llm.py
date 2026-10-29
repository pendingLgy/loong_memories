'''llm 包测试：大模型契约、OpenAI 与兼容厂商实现、注册表与处理器。

覆盖策略：
1. 全方法覆盖：构造方法与 backend、默认值类属性、client、api_key，以及注册表的
   五个公共函数与处理器的 provider、chat、complete。
2. 继承契约验证：BaseLLMProvider 的 model、base_url、temperature、max_tokens、
   api_key 属性与 __init_subclass__ 自动注册行为，均在具体子类实例上验证。
3. 不单独测试私有方法：_target 不做直接断言，其逻辑由 chat 与 provider 的行为间接验证。
4. 不为父类建测试类：BaseLLMProvider 不设独立测试套件，仅通过具体子类验证契约。

全部用例不发起网络请求：client 只验证构造结果，真实 chat 请求由人工联调覆盖。
'''

from __future__ import annotations

import pytest


try:
    import openai

    from conftest import CONFIG_LLM_SETTINGS
    from loong_memories.core.exceptions import ConfigError, LLMError
    from loong_memories.llm.base import BaseLLMProvider
    from loong_memories.llm.deepseek_llm import DeepSeekLLMProvider
    from loong_memories.llm.llm_handler import LLMHandler
    from loong_memories.llm.openai_llm import OpenAILLMProvider
    from loong_memories.llm.qwen_llm import QwenLLMProvider
    from loong_memories.llm.registry_llm import (
        BUILTIN_PROVIDERS,
        available_llm_providers,
        is_llm_registered,
        register_llm,
        register_llm_provider,
        resolve_llm_provider,
    )

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)


# ---------- 替身与辅助 ----------


class _RecordingLLM:
    '''替身 provider：只需实现 chat，记录调用参数并返回固定回复。'''

    def __init__(self, reply='reply'):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **overrides):
        self.calls.append({'messages': messages, 'overrides': overrides})
        return self.reply


class _EchoLLM(BaseLLMProvider):
    '''自定义 provider：声明 backend 即自动注册，同时验证基类属性解析。'''

    backend = 'ut_echo_llm'
    default_model = 'echo-llm-1'
    default_base_url = 'http://echo.local/v1'
    api_key_env = 'UT_ECHO_LLM_KEY'

    def chat(self, messages, **overrides):
        content = messages[-1].get('content') if messages else ''
        return 'echo:' + str(content)


# ---------- fixtures ----------

@pytest.fixture
def settings():
    '''llm 分区配置：取自 conftest 常量，逐例深拷贝避免互相污染。'''
    return CONFIG_LLM_SETTINGS.model_copy(deep=True)


class TestCustomLLMProviderContract:
    '''自定义 provider 的公共契约：继承 BaseLLMProvider 后的属性解析与 chat 分发。'''

    def test_backend_registered_on_subclass_definition(self):
        assert _EchoLLM.backend == 'ut_echo_llm'
        assert is_llm_registered('ut_echo_llm') is True
        assert resolve_llm_provider('ut_echo_llm') is _EchoLLM

    def test_model_prefers_configured_value(self, settings):
        llm = _EchoLLM(settings.model_copy(update={'model': 'custom-llm'}))
        assert llm.model == 'custom-llm'

    def test_model_falls_back_to_default(self, settings):
        llm = _EchoLLM(settings.model_copy(update={'model': ''}))
        assert llm.model == 'echo-llm-1'

    def test_base_url_prefers_configured_value(self, settings):
        llm = _EchoLLM(settings.model_copy(update={'base_url': 'http://gw.local/v2'}))
        assert llm.base_url == 'http://gw.local/v2'

    def test_base_url_falls_back_to_default(self, settings):
        llm = _EchoLLM(settings.model_copy(update={'base_url': ''}))
        assert llm.base_url == 'http://echo.local/v1'

    def test_temperature_reflects_settings(self, settings):
        assert _EchoLLM(settings).temperature == settings.temperature
        assert _EchoLLM(settings.model_copy(update={'temperature': 1.5})).temperature == 1.5

    def test_max_tokens_reflects_settings(self, settings):
        assert _EchoLLM(settings).max_tokens == settings.max_tokens
        assert _EchoLLM(settings.model_copy(update={'max_tokens': 0})).max_tokens == 0

    def test_api_key_prefers_configured_value(self, settings, monkeypatch):
        monkeypatch.setenv('UT_ECHO_LLM_KEY', 'env-key')
        llm = _EchoLLM(settings.model_copy(update={'api_key': 'cfg-key'}))
        assert llm.api_key == 'cfg-key'

    def test_api_key_falls_back_to_env(self, settings, monkeypatch):
        monkeypatch.setenv('UT_ECHO_LLM_KEY', 'env-key')
        llm = _EchoLLM(settings.model_copy(update={'api_key': ''}))
        assert llm.api_key == 'env-key'

    def test_api_key_missing_raises_llm_error(self, settings, monkeypatch):
        monkeypatch.delenv('UT_ECHO_LLM_KEY', raising=False)
        llm = _EchoLLM(settings.model_copy(update={'api_key': ''}))
        with pytest.raises(LLMError) as exc:
            _ = llm.api_key
        assert 'UT_ECHO_LLM_KEY' in str(exc.value)

    def test_chat_returns_subclass_result(self, settings):
        llm = _EchoLLM(settings)
        assert llm.chat([{'role': 'user', 'content': 'hi'}]) == 'echo:hi'


class TestRegistryLLM:
    '''registry_llm 模块的公共函数：登记、查询、解析与内置清单。'''

    def test_builtin_providers_declares_three_vendors(self):
        assert BUILTIN_PROVIDERS['openai'].endswith('OpenAILLMProvider')
        assert BUILTIN_PROVIDERS['qwen'].endswith('QwenLLMProvider')
        assert BUILTIN_PROVIDERS['deepseek'].endswith('DeepSeekLLMProvider')

    def test_available_llm_providers_returns_sorted_names(self):
        names = available_llm_providers()
        assert names == sorted(names)
        assert 'openai' in names and 'qwen' in names and 'deepseek' in names

    def test_is_llm_registered(self):
        assert is_llm_registered('openai') is True
        assert is_llm_registered('not-exist') is False

    def test_resolve_llm_provider_returns_builtin_classes(self):
        assert resolve_llm_provider('openai') is OpenAILLMProvider
        assert resolve_llm_provider('qwen') is QwenLLMProvider
        assert resolve_llm_provider('deepseek') is DeepSeekLLMProvider

    def test_resolve_llm_provider_unknown_name_raises_config_error(self):
        with pytest.raises(ConfigError) as exc:
            resolve_llm_provider('not-exist')
        assert 'not-exist' in str(exc.value)

    def test_register_llm_with_class_object(self):
        class LocalLLM(BaseLLMProvider):
            def chat(self, messages, **overrides):
                return ''

        register_llm('ut_local_llm', LocalLLM)
        assert is_llm_registered('ut_local_llm') is True
        assert resolve_llm_provider('ut_local_llm') is LocalLLM

    def test_register_llm_provider_decorator(self):
        @register_llm_provider('ut_local_llm_decorated')
        class DecorateLLM(BaseLLMProvider):
            def chat(self, messages, **overrides):
                return ''

        assert resolve_llm_provider('ut_local_llm_decorated') is DecorateLLM
        assert available_llm_providers() == sorted(available_llm_providers())


class TestOpenAILLMProvider:
    '''OpenAILLMProvider 的公共契约：backend、默认值类属性与 client 构造。'''

    def test_backend_is_registered(self):
        assert OpenAILLMProvider.backend == 'openai'
        assert is_llm_registered('openai') is True
        assert resolve_llm_provider('openai') is OpenAILLMProvider

    def test_class_defaults(self):
        assert OpenAILLMProvider.default_model == 'gpt-4o-mini'
        assert OpenAILLMProvider.default_base_url == 'https://api.openai.com/v1'
        assert OpenAILLMProvider.api_key_env == 'OPENAI_API_KEY'

    def test_init_does_not_require_api_key(self, settings, monkeypatch):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        llm = OpenAILLMProvider(settings.model_copy(update={'api_key': ''}))
        assert llm.model == settings.model

    def test_client_is_openai_instance(self, settings):
        llm = OpenAILLMProvider(settings)
        assert isinstance(llm.client, openai.OpenAI)
        assert 'api.example.com' in str(llm.client.base_url)


class TestDeepSeekLLMProvider:
    '''DeepSeekLLMProvider 的公共契约：OpenAI 兼容默认值与 API Key 解析。'''

    def test_backend_is_registered(self):
        assert DeepSeekLLMProvider.backend == 'deepseek'
        assert is_llm_registered('deepseek') is True
        assert resolve_llm_provider('deepseek') is DeepSeekLLMProvider
        assert issubclass(DeepSeekLLMProvider, OpenAILLMProvider)

    def test_class_defaults(self):
        assert DeepSeekLLMProvider.default_model == 'deepseek-chat'
        assert 'api.deepseek.com' in DeepSeekLLMProvider.default_base_url
        assert DeepSeekLLMProvider.api_key_env == 'DEEPSEEK_API_KEY'

    def test_api_key_falls_back_to_deepseek_env(self, settings, monkeypatch):
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'env-deepseek-key')
        llm = DeepSeekLLMProvider(settings.model_copy(update={'provider': 'deepseek', 'api_key': ''}))
        assert llm.api_key == 'env-deepseek-key'

    def test_api_key_missing_raises_llm_error(self, settings, monkeypatch):
        monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
        llm = DeepSeekLLMProvider(settings.model_copy(update={'provider': 'deepseek', 'api_key': ''}))
        with pytest.raises(LLMError) as exc:
            _ = llm.api_key
        assert 'DEEPSEEK_API_KEY' in str(exc.value)


class TestQwenLLMProvider:
    '''QwenLLMProvider 的公共契约：DashScope 兼容默认值与 API Key 解析。'''

    def test_backend_is_registered(self):
        assert QwenLLMProvider.backend == 'qwen'
        assert is_llm_registered('qwen') is True
        assert resolve_llm_provider('qwen') is QwenLLMProvider
        assert issubclass(QwenLLMProvider, OpenAILLMProvider)

    def test_class_defaults(self):
        assert QwenLLMProvider.default_model == 'qwen3.7-flash'
        assert 'compatible-mode' in QwenLLMProvider.default_base_url
        assert QwenLLMProvider.api_key_env == 'DASHSCOPE_API_KEY'

    def test_api_key_falls_back_to_dashscope_env(self, settings, monkeypatch):
        monkeypatch.setenv('DASHSCOPE_API_KEY', 'env-qwen-key')
        llm = QwenLLMProvider(settings.model_copy(update={'provider': 'qwen', 'api_key': ''}))
        assert llm.api_key == 'env-qwen-key'

    def test_api_key_missing_raises_llm_error(self, settings, monkeypatch):
        monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
        llm = QwenLLMProvider(settings.model_copy(update={'provider': 'qwen', 'api_key': ''}))
        with pytest.raises(LLMError) as exc:
            _ = llm.api_key
        assert 'DASHSCOPE_API_KEY' in str(exc.value)


class TestLLMHandler:
    '''LLMHandler 的公共契约：provider 装配、chat 透传与 complete 消息构造。'''

    def test_provider_uses_injected_model(self, settings):
        model = _RecordingLLM()
        assert LLMHandler(settings, model).provider is model

    def test_provider_resolves_from_settings(self, settings):
        handler = LLMHandler(settings)
        assert isinstance(handler.provider, DeepSeekLLMProvider)

    def test_provider_is_cached(self, settings):
        handler = LLMHandler(settings)
        assert handler.provider is handler.provider

    def test_chat_passes_messages_and_overrides(self, settings):
        model = _RecordingLLM('reply')
        messages = [{'role': 'user', 'content': 'hi'}]
        result = LLMHandler(settings, model).chat(messages, temperature=0.1)
        assert result == 'reply'
        assert model.calls[0]['messages'] is messages
        assert model.calls[0]['overrides'] == {'temperature': 0.1}

    def test_complete_sends_single_user_message(self, settings):
        model = _RecordingLLM('done')
        assert LLMHandler(settings, model).complete('hello') == 'done'
        assert model.calls[0]['messages'] == [{'role': 'user', 'content': 'hello'}]

    def test_complete_prepends_system_message(self, settings):
        model = _RecordingLLM()
        LLMHandler(settings, model).complete('hello', system='be brief')
        assert model.calls[0]['messages'] == [
            {'role': 'system', 'content': 'be brief'},
            {'role': 'user', 'content': 'hello'},
        ]

    def test_complete_passes_overrides(self, settings):
        model = _RecordingLLM()
        LLMHandler(settings, model).complete('hello', max_tokens=8)
        assert model.calls[0]['overrides'] == {'max_tokens': 8}

    def test_chat_dispatches_to_registered_provider(self, settings):
        handler = LLMHandler(settings.model_copy(update={'provider': 'ut_echo_llm', 'model': ''}))
        assert handler.chat([{'role': 'user', 'content': 'hi'}]) == 'echo:hi'
