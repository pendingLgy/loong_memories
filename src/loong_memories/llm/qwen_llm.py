# 通义千问（DashScope 兼容模式）大模型实现，沿用 OpenAI 请求格式。
#
# 与 OpenAI 仅默认模型、接口地址与 API Key 环境变量不同，故直接继承。

from __future__ import annotations

from loong_memories.llm.openai_llm import OpenAILLMProvider


class QwenLLMProvider(OpenAILLMProvider):
    """调用 DashScope 兼容接口；默认模型与端点属类属性，可按部署环境覆盖。"""

    backend = "qwen"
    default_model = "qwen3.7-plus"
    default_base_url = "https://llm-aa9vce2460xvbj3o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"

