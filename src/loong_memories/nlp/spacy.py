from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "zh_core_web_md"

# 失败哨兵：与「尚未加载」区分，避免每次调用都重复尝试加载。
_FAILED = object()
_PIPELINES: Dict[str, Any] = {}
_PIPELINES_LEMMA: Dict[str, Any] = {}
_lock = threading.Lock()


def load_nlp(model_name: str = DEFAULT_MODEL) -> Optional[Any]:
    """加载 spaCy 句柄：补 sentencizer 以支持句级切分；未装 spaCy 或模型缺失时返回 None。"""
    cached = _PIPELINES.get(model_name)
    if cached is _FAILED:
        return None
    if cached is not None:
        return cached

    #
    with _lock:
        # 双重检查（Double-Checked Locking）
        cached = _PIPELINES.get(model_name)
        if cached is _FAILED:
            return None
        if cached is not None:
            return cached

        try:
            import spacy
        except ImportError:
            _PIPELINES[model_name] = _FAILED
            logger.warning("spaCy 未安装，分句与实体抽取将使用回退实现 model=%s", model_name)
            return None

        try:
            nlp = spacy.load(model_name)
        except Exception as exc:  # noqa: BLE001 模型缺失或加载失败均降级
            _PIPELINES[model_name] = _FAILED
            logger.warning("spaCy 模型加载失败，将使用回退实现 model=%s error=%s", model_name, exc)
            return None

        ensure_sentence_boundaries(nlp)
        _PIPELINES[model_name] = nlp
        logger.info("spaCy 模型加载完成 model=%s pipes=%s", model_name, list(nlp.pipe_names))
    return nlp

def get_nlp_lemma(model_name: str = DEFAULT_MODEL) -> Optional[Any]:
    """Return spaCy model with only lemmatizer for BM25 text processing."""
    cached = _PIPELINES_LEMMA.get(model_name)
    if cached is _FAILED:
        return None
    if cached is not None:
        return cached

    #
    with _lock:
        # 双重检查（Double-Checked Locking）
        cached = _PIPELINES_LEMMA.get(model_name)
        if cached is _FAILED:
            return None
        if cached is not None:
            return cached

        try:
            import spacy
        except ImportError:
            _PIPELINES_LEMMA[model_name] = _FAILED
            logger.warning("spaCy lemma 未安装，分句与实体抽取将使用回退实现 model=%s", model_name)
            return None

        try:
            nlp_lemma = spacy.load(model_name, disable=["ner", "parser"])
        except Exception as exc:  # noqa: BLE001 模型缺失或加载失败均降级
            _PIPELINES_LEMMA[model_name] = _FAILED
            logger.warning("spaCy lemma 模型加载失败，将使用回退实现 model=%s error=%s", model_name, exc)
            return None

        _PIPELINES_LEMMA[model_name] = nlp_lemma
        logger.info("spaCy lemma 模型加载完成 model=%s pipes=%s", model_name, list(nlp_lemma.pipe_names))
    return nlp_lemma


def clear_cache() -> None:
    """清空句柄缓存（含失败记录），供测试或运行时重建句柄使用。"""
    with _lock:  # 【建议】清空缓存时也加上锁，确保多线程安全
        _PIPELINES.clear()


def ensure_sentence_boundaries(nlp: Any) -> None:
    """句级切分依赖 parser 或 sentencizer；两者都缺时补一个 sentencizer。"""
    names = list(getattr(nlp, "pipe_names", None) or [])
    if "parser" not in names and "sentencizer" not in names:
        nlp.add_pipe("sentencizer")