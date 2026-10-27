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
 DeepSeek 大模型实现，沿用 OpenAI 请求格式。
 与 OpenAI 仅默认模型、接口地址与 API Key 环境变量不同，故直接继承。
"""


from __future__ import annotations

from loong_memories.llm.openai_llm import OpenAILLMProvider


class DeepSeekLLMProvider(OpenAILLMProvider):
    """调用 DeepSeek 开放平台；默认模型与端点属类属性，可按部署环境覆盖。"""

    backend = "deepseek"
    default_model = "deepseek-chat"
    default_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"

