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
 大模型注册表：支持用户注册自定义 LLM 实现，只负责注册与解析，不做实例化。

 注册方式二选一：继承 BaseLLMProvider 并声明 backend（自动注册），或调用 register_llm 显式登记。
 注册值可为类对象，也可为点分路径（首次解析时惰性导入）。
 实例化在 LLMHandler 调用时进行，本模块不持有任何实例。
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Type

from loong_memories.registry import Ref, Registry

if TYPE_CHECKING:  # 仅在类型检查期引用，避免与 base 形成导入环
    from loong_memories.llm.base import BaseLLMProvider

# 内置 provider 名称到类路径的映射，按需惰性导入。
BUILTIN_PROVIDERS: Dict[str, str] = {
    "openai": "loong_memories.llm.openai_llm.OpenAILLMProvider",
    "qwen": "loong_memories.llm.qwen_llm.QwenLLMProvider",
    "deepseek": "loong_memories.llm.deepseek_llm.DeepSeekLLMProvider",
}

_REGISTRY = Registry("大模型", BUILTIN_PROVIDERS)


def register_llm(name: str, ref: Ref) -> None:
    "登记大模型：ref 可为类对象，也可为点分路径字符串。"
    _REGISTRY.register(name, ref)


def register_llm_provider(name: str):
    "装饰器形式登记大模型。"
    return _REGISTRY.decorator(name)


def is_llm_registered(name: str) -> bool:
    "判断大模型是否已登记。"
    return name in _REGISTRY


def available_llm_providers() -> List[str]:
    "返回已登记的 provider 名，按字典序排列。"
    return _REGISTRY.names()


def resolve_llm_provider(name: str) -> Type[BaseLLMProvider]:
    "按 provider 名取得实现类，路径登记的在此惰性导入。"
    return _REGISTRY.resolve(name)
