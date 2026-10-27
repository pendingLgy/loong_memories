# DeepSeek 大模型实现，沿用 OpenAI 请求格式。
#
# 与 OpenAI 仅默认模型、接口地址与 API Key 环境变量不同，故直接继承。

from __future__ import annotations

from loong_memories.llm.openai_llm import OpenAILLMProvider


class DeepSeekLLMProvider(OpenAILLMProvider):
    """调用 DeepSeek 开放平台；默认模型与端点属类属性，可按部署环境覆盖。"""

    backend = "deepseek"
    default_model = "deepseek-chat"
    default_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"

