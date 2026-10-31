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


"""大模型相关子配置：embedding（向量化）、llm（对话与事实抽取）、rerank（检索结果重排）。

三个模型都是 BaseModel 而非 BaseSettings：字段由 core.loader_config 统一装载后校验，
本模块只声明字段语义与取值约束，不持有任何运行时状态。

embedding 与 rerank 的配置归档在本文件是历史约定，三者服务不同组件：
embedding 供 embedder 使用，llm 供事实抽取与摘要使用，rerank 供检索结果重排使用。
"""

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingSettings(BaseModel):
    """向量化参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # provider 名对应嵌入模型注册表中的键；继承 BaseEmbeddingProvider 即自动注册，
    # 内置 openai 与 qwen，可用 .available_embedding_providers() 查看。
    provider: str = Field(default="openai")
    # 模型名。留空时用 provider 的默认模型，见各 provider 的 default_model。
    model: str = Field(default="")
    batch_size: int = Field(default=128, gt=0)
    # 期望输出维度；向量化结果按此校验，默认与内置 qwen 模型一致。
    dim: int = Field(default=1024, gt=0, description="向量维度，需与模型实际维度及 vector.vector_size 一致")
    # API Key。留空时读取 provider 约定的环境变量，如 OPENAI_API_KEY、DASHSCOPE_API_KEY。
    api_key: str = Field(default="")
    # 接口地址。留空时用 provider 默认地址，可指向自建网关或代理。
    base_url: str = Field(default="")
    # 必须为正数：0 或负数会让底层 HTTP 客户端立即超时，嵌入调用必然失败。
    timeout: float = Field(default=60.0, gt=0, description="单次嵌入请求的超时时间（秒），必须大于 0")


class LLMSettings(BaseModel):
    """大模型参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化对应组件。
    enable: bool = Field(default=False)

    # provider 名对应大模型注册表中的键；内置 openai、qwen、deepseek，qwen 与 deepseek 兼容 OpenAI。
    provider: str = Field(default="openai")
    # 模型名。留空时用 provider 的默认模型，见各 provider 的 default_model。
    model: str = Field(default="")
    # 采样温度，范围 0 到 2。
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # 单次生成的最大 token 数，0 表示不显式传入（由服务端决定）。
    max_tokens: int = Field(default=0, ge=0)
    # API Key。留空时读取 provider 约定的环境变量，如 OPENAI_API_KEY、DASHSCOPE_API_KEY、DEEPSEEK_API_KEY。
    api_key: str = Field(default="")
    # 接口地址。留空时用 provider 默认地址，可指向自建网关或代理。
    base_url: str = Field(default="")
    timeout: float = Field(default=60.0, gt=0, description="单次对话请求的超时时间（秒），必须大于 0")


class RerankSettings(BaseModel):
    """重排参数：provider 决定实现，其余为通用与实现专属配置。"""

    model_config = ConfigDict(extra="forbid")

    # 是否启用该分区；为 False 时门面不初始化 reranker，检索直接使用融合打分结果。
    enable: bool = Field(default=False, description="是否启用重排分区")

    # provider 名对应 rerank 注册表中的键；继承 BaseRerankProvider 并声明 backend 即自动注册，
    # 内置 qwen，可用 available_rerank_providers() 查看。
    provider: str = Field(default="qwen", description="重排 provider 名，需在 rerank 注册表中存在")

    # 模型名。留空时用 provider 的默认模型，见各 provider 的 default_model。
    model: str = Field(default="", description="模型名，留空则使用 provider 默认模型")

    # 默认返回条数；调用方显式传入 top_k 时以调用方为准。
    top_k: int = Field(default=5, gt=0, description="默认返回条数，调用方传入 top_k 时以调用方为准")

    # 兼容保留：DashScope text-rerank 接口不接受该参数，当前配置不生效。
    max_tokens: int = Field(default=0, ge=0, description="单次生成上限 token 数；text-rerank 接口不消费")

    # 单次请求参与重排的最大文档数，超出部分截断，避免请求体过大。
    max_documents: int = Field(default=100, gt=0, description="单次请求参与重排的最大文档数")

    # API Key。留空时读取 provider 约定的环境变量，如 DASHSCOPE_API_KEY。
    api_key: str = Field(default="", description="API Key，留空则读取 provider 约定的环境变量")

    # 接口地址。留空时用 provider 默认地址，可指向自建网关或代理。
    base_url: str = Field(default="", description="接口地址，留空则使用 provider 默认地址")

    # 请求超时（秒）。必须为正数，避免 requests 拿到 0 时立即超时。
    timeout: float = Field(default=60.0, gt=0, description="单次重排请求的超时时间（秒），必须大于 0")
