# -*- coding: utf-8 -*-
"""
这个文件干什么：建向量库（存）+ 检索（查）。

【向量库到底存了什么】
一个三元组，按 id 塞进一个字典：

    id: "a3f1"
     ├─ vector    [0.12, -0.87, ..., 0.05]   ← 用于比距离
     ├─ text      "【12-补充-检索与RAG.md】... HNSW 建了一个多层图"
     └─ metadata  {"source": ..., "title_path": ...}   ← 过滤 / 展示引用

InMemoryVectorStore 就这点东西 —— 一个 dict 加一个 numpy 矩阵。

【为什么先用它】
numpy 之外零依赖，而且接口和 Chroma 完全一致 —— M5 换库时只改构造那一行，
调用 search 的代码一个字不动。这就是走 LangChain 的收益点。
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

import config
from embedder import get_embeddings


def build_store(docs: list[Document]) -> InMemoryVectorStore:
    """把一堆 Document 向量化，装进内存向量库。"""
    return InMemoryVectorStore.from_documents(docs, embedding=get_embeddings())
    # ↑ from_documents 是个类方法：先造空 store 再 add_documents，纯语法糖。
    #   已有库要追加就用 store.add_documents([...]) —— "笔记更新了怎么办"的答案在这。


def search(
    store: InMemoryVectorStore, query: str, k: int | None = None
) -> list[tuple[Document, float]]:
    """检索，返回 [(Document, 分数), ...]，分数越大越像。"""
    return store.similarity_search_with_score(query, k=k or config.TOP_K)
    # ↑ 内部四步：
    #   ① embed_query(query)                       ← 用同一个模型翻成 1024 个数
    #   ② cosine_similarity(查询向量, 全部块向量)   ← numpy 一次算完
    #   ③ argsort()[::-1]                           ← 按分数降序
    #   ④ 取前 k 个，连原文 + metadata 一起还
    #
    # 用 with_score 而不是 similarity_search：后者不带分数。
    # 调检索效果时分数是唯一的反馈信号，看不见分数就只能瞎猜。