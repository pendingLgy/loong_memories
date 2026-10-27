"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging
import re

from loong_memories.nlp.spacy import get_nlp_lemma

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """

    nlp_lemma = get_nlp_lemma()
    if nlp_lemma is None:
        return text

    doc = nlp_lemma(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)


def lemmatize_zh_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching. adapter milvus BM25

    Returns space-joined lemmas for full-text search. Handles both split
    and un-split technical identifiers using underscores and hyphens (e.g., 'vcl_c', 'Chapter_1', 'rag-data').
    """
    nlp_lemma = get_nlp_lemma()
    if nlp_lemma is None:
        return text

    doc = nlp_lemma(text)

    # 1. 纯连接符正则（仅保留下划线 _ 和标准连字符 -）
    connector_pattern = re.compile(r'^[-_]$')
    alphanumeric_pattern = re.compile(r'^[a-zA-Z0-9]+$')

    # 2. 复合技术标识符判定
    def is_tech_identifier(w: str) -> bool:
        has_content = bool(re.search(r'[a-zA-Z0-9\u4e00-\u9fa5]', w))
        has_connector = bool(re.search(r'[-_]', w))
        return has_content or has_connector

    raw_tokens = []
    for token in doc:
        word = token.text

        # 1. 如果既是停用词又是标点符号，直接 continue
        if token.is_stop and token.is_punct and not bool(connector_pattern.match(word)):
            continue

        # 2. 如果是停用词，反向查找上一个有效的文本字符
        if token.is_stop:
            if not bool(alphanumeric_pattern.match(word)) and not bool(connector_pattern.match(word)):
                continue

            prev_idx = len(raw_tokens) - 1
            while prev_idx >= 0 and bool(connector_pattern.match(raw_tokens[prev_idx])):
                prev_idx -= 1

            if prev_idx < 0:
                continue

        # 3. 过滤掉其他非连接符的普通标点
        if token.is_punct and not bool(connector_pattern.match(word)):
            continue

        if (word.isalnum()
                or word.endswith("ing")
                or bool(connector_pattern.match(word))
                or is_tech_identifier(word)):
            raw_tokens.append(word)

    #  结构化的前向扫描与组合拼接逻辑
    tokens = []
    i = 0
    while i < len(raw_tokens):
        current = raw_tokens[i]

        # 检查是否可以通过连接符与后面的词组成复合词（如 Chapter _ 1）
        if i + 2 < len(raw_tokens) and connector_pattern.match(raw_tokens[i + 1]):
            connector = raw_tokens[i + 1]
            right_token = raw_tokens[i + 2]
            candidate = f"{current}{connector}{right_token}"

            # 校验原文中是否真实包含该复合词
            if candidate in text:
                # 严格按照预期顺序依次添加：左词、右词、完整复合词
                tokens.append(current)
                tokens.append(right_token)
                tokens.append(candidate)
                i += 3
                continue

        # 处理本身包含下划线/连字符的单一 token (未被切分的复合标识符，如 vibe_cli)
        if is_tech_identifier(current) and re.search(r'[-_]', current):
            sub_parts = re.split(r'[-_]', current)
            for part in sub_parts:
                if part:
                    tokens.append(part)

            # 过滤掉单个连接符
            if len(current.strip()) == 1 and bool(connector_pattern.match(current)):
                i += 1
                continue
            else:
                tokens.append(current)
                i += 1
                continue

        # 普通文本 Token
        tokens.append(current)
        i += 1

    return " ".join(tokens)
