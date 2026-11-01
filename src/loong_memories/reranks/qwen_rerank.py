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


"""DashScope 原生 Text-Rerank 接口实现。

只负责一次重排调用与结果回填：正文提取、请求组装、分数回填、失败降级。
接口不可用时保留原始顺序并给出中立分值，避免上游检索链路因重排故障中断。
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from loong_memories.reranks.base import BaseRerankProvider

logger = logging.getLogger(__name__)

# 降级时赋予的中立分值，与接口相关性分数同处 0 到 1 区间。
FALLBACK_SCORE = 0.5



class QwenReranker(BaseRerankProvider):
    """基于阿里云百炼 (DashScope) 原生 Text-Rerank 接口的重排实现。"""

    backend = "qwen"
    default_model = "qwen3.7-text-rerank"
    default_base_url = "https://llm-aa9vce2460xvbj3o.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    api_key_env = "DASHSCOPE_API_KEY"

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """对候选文档重排，返回按相关性降序、携带重排分数的文档副本。

        query 为空时无法重排，直接按原顺序降级返回。
        documents 的元素可以是字典，也可以是带 memory 或 text 属性的对象，一律按副本处理。
        top_k 为返回条数上限，None 表示不截断，小于等于 0 表示不返回。
        接口失败不抛异常，按原顺序返回并标记 rerank_fallback。
        """
        if not documents:
            return documents

        # 1. 提取每个文档的文本内容（兼容 memory, text, content 或纯字符串）
        doc_texts = []
        for doc in documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
            elif isinstance(doc, dict):
                if 'memory' in doc:
                    doc_texts.append(doc['memory'])
                elif 'text' in doc:
                    doc_texts.append(doc['text'])
                elif 'content' in doc:
                    doc_texts.append(doc['content'])
                else:
                    doc_texts.append(str(doc))
            else:
                doc_texts.append(str(doc))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_doc_texts = doc_texts[:self.max_documents]

        # 确定最终的 top_n
        effective_top_k = top_k
        top_n = effective_top_k if effective_top_k is not None else len(max_doc_texts)

        # 2. 组装符合 Qwen Rerank API 规范的请求体
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": max_doc_texts,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": True,
                "instruct": "Retrieve semantically similar text.",
            }
        }

        response = None
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            res_json = response.json()

            # 4. 解析接口返回的排序结果
            results = res_json.get("output", {}).get("results", [])

            scored_docs = []
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score", 0.0)

                original_doc = documents[idx]
                if isinstance(original_doc, dict):
                    scored_doc = original_doc.copy()
                else:
                    scored_doc = {"text": original_doc}

                if scored_doc.get("score_details"):
                    scored_doc.get("score_details")['rerank_score'] = score

                scored_doc['score'] = score

                scored_docs.append(scored_doc)

            # 5. 按评分降序排序（Qwen 接口本身已排序，这里做双重保障）
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            return scored_docs

        except Exception as e:
            # 降级容错：若请求失败，记录日志并赋予中立分值返回原文档
            if response:
                logger.error("http response:%s",response.text)

            logger.error("Qwen reranking failed, msg: %s, falling back to original documents order:", e)
            fallback_docs = []
            for doc in documents:
                if isinstance(doc, dict):
                    d = doc.copy()
                    d['score'] = FALLBACK_SCORE
                else:
                    d = {"text": doc, 'score': FALLBACK_SCORE}

                    if d.get("score_details"):
                        d.get("score_details")['rerank_score'] = FALLBACK_SCORE
                        d.get("score_details")['rerank_fail'] = True

                fallback_docs.append(d)

            if effective_top_k:
                fallback_docs = fallback_docs[:effective_top_k]
            return fallback_docs