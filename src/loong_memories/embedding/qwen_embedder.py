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
 通义千问（DashScope 兼容模式）嵌入实现（embedder），沿用 OpenAI 请求格式。

 与 OpenAI 仅默认模型、接口地址、API Key 环境变量与批量上限不同，故直接继承。
"""


from __future__ import annotations

from loong_memories.embedding.openai_embedder import OpenAIEmbedder


class QwenEmbedder(OpenAIEmbedder):
    """调用 DashScope 兼容接口；默认模型与端点见类属性，可按部署环境覆盖。"""

    backend = "qwen"
    default_model = "qwen3.7-text-embedding"
    default_dim = 1024
    default_base_url = "https://llm-aa9vce2460xvbj3o.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    api_key_env = "DASHSCOPE_API_KEY"
    # DashScope 单次请求最多接受 10 条文本，超限会被拒绝。
    max_batch_size = 10
