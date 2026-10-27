'''embedding 包测试：嵌入模型契约、OpenAI 与 Qwen 实现、注册表与处理器。

覆盖策略：
1. 全方法覆盖：构造方法与 backend、默认值类属性、client、api_key，以及注册表的
   五个公共函数与处理器的 provider、encode。
2. 继承契约验证：BaseEmbedder 的 model、base_url、dim、api_key 属性与
   __init_subclass__ 自动注册行为，均在具体子类实例上验证。
3. 不单独测试私有方法：_target、_batch_size、_encode_batch、_validate_dim
   不做直接断言，其逻辑由 encode 与 provider 的行为间接验证。
4. 不为父类建测试类：BaseEmbedder 不设独立测试套件，仅通过具体子类验证契约。

全部用例不发起网络请求：client 只验证构造结果，真实 encode 请求由人工联调覆盖。
'''

from __future__ import annotations

import pytest


try:
    import openai

    from conftest import CONFIG_EMBEDDING_SETTINGS
    from loong_memories.core.exceptions import ConfigError, EmbeddingError
    from loong_memories.embedding.base import BaseEmbedder
    from loong_memories.embedding.embedder_handler import EmbedderHandler
    from loong_memories.embedding.openai_embedder import OpenAIEmbedder
    from loong_memories.embedding.qwen_embedder import QwenEmbedder
    from loong_memories.embedding.registry_embedder import (
        BUILTIN_EMBEDDERS,
        available_embedders,
        is_embedder_registered,
        register_embedder,
        register_embedder_provider,
        resolve_embedder,
    )

except ImportError as _exc:
    pytest.skip(f'[SKIP INFO] File {__file__}， 缺少 {str(_exc)}', allow_module_level=True)


# ---------- 替身与辅助 ----------


class _RecordingModel:
    '''替身 embedder：只需实现 encode，可选 max_batch_size。'''

    def __init__(self, dim, max_batch_size=0, vector_factory=None):
        self.dim = dim
        self.max_batch_size = max_batch_size
        self.vector_factory = vector_factory
        self.batches = []

    def encode(self, texts):
        self.batches.append(list(texts))
        if self.vector_factory is not None:
            return [self.vector_factory(text) for text in texts]
        return [[1.0] * self.dim for _ in texts]


class _EchoEmbedder(BaseEmbedder):
    '''自定义 embedder：声明 backend 即自动注册，同时验证基类属性解析。'''

    backend = 'ut_echo'
    default_model = 'echo-embedding-1'
    default_dim = 4
    default_base_url = 'http://echo.local/v1'
    api_key_env = 'UT_ECHO_API_KEY'
    max_batch_size = 2

    def encode(self, texts):
        return [[float(len(text))] * self._settings.dim for text in texts]


# ---------- fixtures ----------

@pytest.fixture
def settings():
    '''embedding 分区配置：取自 conftest 常量，逐例深拷贝避免互相污染。'''
    return CONFIG_EMBEDDING_SETTINGS.model_copy(deep=True)


class TestCustomEmbedderContract:
    '''自定义 embedder 的公共契约：继承 BaseEmbedder 后的属性解析与 encode 分发。'''

    def test_backend_registered_on_subclass_definition(self):
        assert _EchoEmbedder.backend == 'ut_echo'
        assert is_embedder_registered('ut_echo') is True
        assert resolve_embedder('ut_echo') is _EchoEmbedder

    def test_model_prefers_configured_value(self, settings):
        emb = _EchoEmbedder(settings.model_copy(update={'model': 'custom-embed'}))
        assert emb.model == 'custom-embed'

    def test_model_falls_back_to_default(self, settings):
        emb = _EchoEmbedder(settings.model_copy(update={'model': ''}))
        assert emb.model == 'echo-embedding-1'

    def test_base_url_prefers_configured_value(self, settings):
        emb = _EchoEmbedder(settings.model_copy(update={'base_url': 'http://gw.local/v2'}))
        assert emb.base_url == 'http://gw.local/v2'

    def test_base_url_falls_back_to_default(self, settings):
        emb = _EchoEmbedder(settings.model_copy(update={'base_url': ''}))
        assert emb.base_url == 'http://echo.local/v1'

    def test_dim_reflects_settings(self, settings):
        assert _EchoEmbedder(settings).dim == settings.dim
        assert _EchoEmbedder(settings.model_copy(update={'dim': 256})).dim == 256

    def test_api_key_prefers_configured_value(self, settings, monkeypatch):
        monkeypatch.setenv('UT_ECHO_API_KEY', 'env-key')
        emb = _EchoEmbedder(settings.model_copy(update={'api_key': 'cfg-key'}))
        assert emb.api_key == 'cfg-key'

    def test_api_key_falls_back_to_env(self, settings, monkeypatch):
        monkeypatch.setenv('UT_ECHO_API_KEY', 'env-key')
        emb = _EchoEmbedder(settings.model_copy(update={'api_key': ''}))
        assert emb.api_key == 'env-key'

    def test_api_key_missing_raises_embedding_error(self, settings, monkeypatch):
        monkeypatch.delenv('UT_ECHO_API_KEY', raising=False)
        emb = _EchoEmbedder(settings.model_copy(update={'api_key': ''}))
        with pytest.raises(EmbeddingError) as exc:
            _ = emb.api_key
        assert 'UT_ECHO_API_KEY' in str(exc.value)

    def test_encode_returns_vector_per_text_with_settings_dim(self, settings):
        emb = _EchoEmbedder(settings.model_copy(update={'dim': 4}))
        assert emb.encode(['ab', 'abcd']) == [[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0]]


class TestRegistryEmbedder:
    '''registry_embedder 模块的公共函数：登记、查询、解析与内置清单。'''

    def test_builtin_embedders_declares_openai_and_qwen(self):
        assert BUILTIN_EMBEDDERS['openai'].endswith('OpenAIEmbedder')
        assert BUILTIN_EMBEDDERS['qwen'].endswith('QwenEmbedder')

    def test_available_embedders_returns_sorted_names(self):
        names = available_embedders()
        assert names == sorted(names)
        assert 'openai' in names and 'qwen' in names

    def test_is_embedder_registered(self):
        assert is_embedder_registered('openai') is True
        assert is_embedder_registered('not-exist') is False

    def test_resolve_embedder_returns_builtin_classes(self):
        assert resolve_embedder('openai') is OpenAIEmbedder
        assert resolve_embedder('qwen') is QwenEmbedder

    def test_resolve_embedder_unknown_name_raises_config_error(self):
        with pytest.raises(ConfigError) as exc:
            resolve_embedder('not-exist')
        assert 'not-exist' in str(exc.value)

    def test_register_embedder_with_class_object(self):
        class LocalEmbedder(BaseEmbedder):
            def encode(self, texts):
                return []

        register_embedder('ut_local_class', LocalEmbedder)
        assert is_embedder_registered('ut_local_class') is True
        assert resolve_embedder('ut_local_class') is LocalEmbedder

    def test_register_embedder_provider_decorator(self):
        @register_embedder_provider('ut_local_decorated')
        class DecorateEmbedder(BaseEmbedder):
            def encode(self, texts):
                return []

        assert resolve_embedder('ut_local_decorated') is DecorateEmbedder
        assert available_embedders() == sorted(available_embedders())


class TestOpenAIEmbedder:
    '''OpenAIEmbedder 的公共契约：backend、默认值类属性与 client 构造。'''

    def test_backend_is_registered(self):
        assert OpenAIEmbedder.backend == 'openai'
        assert is_embedder_registered('openai') is True
        assert resolve_embedder('openai') is OpenAIEmbedder

    def test_class_defaults(self):
        assert OpenAIEmbedder.default_model == 'text-embedding-3-small'
        assert OpenAIEmbedder.default_dim == 1536
        assert OpenAIEmbedder.default_base_url == 'https://api.openai.com/v1'
        assert OpenAIEmbedder.api_key_env == 'OPENAI_API_KEY'
        assert OpenAIEmbedder.max_batch_size == 0

    def test_init_does_not_require_api_key(self, settings, monkeypatch):
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        emb = OpenAIEmbedder(settings.model_copy(update={'api_key': ''}))
        assert emb.model == settings.model

    def test_client_is_openai_instance(self, settings):
        emb = OpenAIEmbedder(settings)
        assert isinstance(emb.client, openai.OpenAI)
        assert 'api.example.com' in str(emb.client.base_url)


class TestQwenEmbedder:
    '''QwenEmbedder 的公共契约：DashScope 默认值继承与 API Key 解析。'''

    def test_backend_is_registered(self):
        assert QwenEmbedder.backend == 'qwen'
        assert is_embedder_registered('qwen') is True
        assert resolve_embedder('qwen') is QwenEmbedder
        assert issubclass(QwenEmbedder, OpenAIEmbedder)

    def test_class_defaults(self):
        assert QwenEmbedder.default_model == 'qwen3.7-text-embedding'
        assert QwenEmbedder.default_dim == 1024
        assert QwenEmbedder.api_key_env == 'DASHSCOPE_API_KEY'
        assert QwenEmbedder.max_batch_size == 10
        assert 'compatible-mode' in QwenEmbedder.default_base_url

    def test_api_key_falls_back_to_dashscope_env(self, settings, monkeypatch):
        monkeypatch.setenv('DASHSCOPE_API_KEY', 'env-qwen-key')
        emb = QwenEmbedder(settings.model_copy(update={'provider': 'qwen', 'api_key': ''}))
        assert emb.api_key == 'env-qwen-key'

    def test_api_key_missing_raises_embedding_error(self, settings, monkeypatch):
        monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
        emb = QwenEmbedder(settings.model_copy(update={'provider': 'qwen', 'api_key': ''}))
        with pytest.raises(EmbeddingError) as exc:
            _ = emb.api_key
        assert 'DASHSCOPE_API_KEY' in str(exc.value)


class TestEmbedderHandler:
    '''EmbedderHandler 的公共契约：provider 装配、分批编码与维度校验。'''

    def test_provider_uses_injected_model(self, settings):
        model = _RecordingModel(dim=settings.dim)
        assert EmbedderHandler(settings, model).provider is model

    def test_provider_resolves_from_settings(self, settings):
        handler = EmbedderHandler(settings)
        assert isinstance(handler.provider, OpenAIEmbedder)

    def test_provider_is_cached(self, settings):
        handler = EmbedderHandler(settings)
        assert handler.provider is handler.provider

    def test_encode_returns_vector_per_text(self, settings):
        model = _RecordingModel(dim=settings.dim)
        assert len(EmbedderHandler(settings, model).encode(['a', 'b', 'c'])) == 3

    def test_encode_empty_texts_skips_model(self, settings):
        model = _RecordingModel(dim=settings.dim)
        assert EmbedderHandler(settings, model).encode([]) == []
        assert model.batches == []

    def test_encode_splits_by_settings_batch_size(self, settings):
        model = _RecordingModel(dim=settings.dim)
        EmbedderHandler(settings, model).encode(['t'] * 70)
        assert [len(batch) for batch in model.batches] == [settings.batch_size, 70 - settings.batch_size]

    def test_encode_respects_embedder_batch_limit(self, settings):
        model = _RecordingModel(dim=settings.dim, max_batch_size=10)
        EmbedderHandler(settings, model).encode(['t'] * 25)
        assert [len(batch) for batch in model.batches] == [10, 10, 5]

    def test_encode_converts_values_to_float(self, settings):
        model = _RecordingModel(dim=3, vector_factory=lambda text: (1, 2, 3))
        vectors = EmbedderHandler(settings.model_copy(update={'dim': 3}), model).encode(['a'])
        assert vectors == [[1.0, 2.0, 3.0]]
        assert all(isinstance(value, float) for value in vectors[0])

    def test_encode_raises_on_dim_mismatch(self, settings):
        model = _RecordingModel(dim=2)
        with pytest.raises(EmbeddingError) as exc:
            EmbedderHandler(settings, model).encode(['a'])
        assert '向量维度不一致' in str(exc.value)

    def test_encode_dispatches_to_registered_embedder(self, settings):
        handler = EmbedderHandler(settings.model_copy(update={'provider': 'ut_echo', 'model': ''}))
        vectors = handler.encode(['a', 'b'])
        assert vectors == [[1.0] * settings.dim, [1.0] * settings.dim]
