# -*- coding: utf-8 -*-
"""
这个文件干什么：建向量库（存）+ 检索（查）+ 持久化（存到磁盘，重启不用重新向量化）。

【向量库到底存了什么】
一个三元组，按 id 塞进一个字典：

    id: "a3f1"
     ├─ vector    [0.12, -0.87, ..., 0.05]   ← 用于比距离
     ├─ text      "【12-补充-检索与RAG.md】... HNSW 建了一个多层图"
     └─ metadata  {"source": ..., "title_path": ...}   ← 过滤 / 展示引用

以前用 InMemoryVectorStore（一个 dict 加一个 numpy 矩阵），接口和 Chroma
完全一致 —— 这就是当初先用它的原因：换库时只改构造那一行，调用 search
的代码一个字不动。现在这一步兑现了。

【指纹机制：怎么判断"要不要重新向量化"】
每次启动都重新向量化全部 chunk = 每次都调几十次 embedding API、
等几十秒、花一遍钱，但语料大概率没变过。解决办法：给当前这批 chunk
内容算一个哈希（"指纹"），存一份在索引目录里；下次启动重新算一遍指纹，
两个一样就说明内容没变，直接读盘加载，不再调 embedding API；不一样就
整体重建（不做增量更新 —— 当前语料规模下整体重建也就几十秒，犯不上
为增量更新增加复杂度）。

指纹算在"切分后的 chunk"上而不是"切分前的原文"上：这样不仅能感知
笔记内容变了，连以后调整 chunker.py 的切分参数导致切出来的块变了，
也能被感知到 —— 覆盖更全，而且切分这一步不联网、很快，不增加成本。
"""
from __future__ import annotations

import hashlib
import pickle

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config
from embedder import get_embeddings


FINGERPRINT_FILE = "fingerprint.txt"
BM25_INDEX_FILE = "bm25.pkl"  # BM25 索引序列化文件


def _tokenize(text: str) -> list[str]:
    """分词：中文按字符切分，保留字母数字（用于术语如 HNSW、BM25）。

    例："HNSW 的 M 参数" -> ["HNSW", "的", "M", "参数"]
    """
    tokens = []
    current = []
    for char in text:
        if char.isalnum():  # 字母或数字
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if not char.isspace():  # 中文字符（非空白）
                tokens.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _build_bm25_index(chunks: list[Document]) -> BM25Okapi:
    """构建 BM25 索引：把每个 chunk 的文本分词，喂给 BM25Okapi。"""
    tokenized_corpus = [_tokenize(doc.page_content) for doc in chunks]
    return BM25Okapi(tokenized_corpus)


def _save_bm25_index(bm25: BM25Okapi) -> None:
    """把 BM25 索引序列化到磁盘（pickle）。"""
    path = config.INDEX_DIR / BM25_INDEX_FILE
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def _load_bm25_index() -> BM25Okapi:
    """从磁盘加载 BM25 索引（pickle）。"""
    path = config.INDEX_DIR / BM25_INDEX_FILE
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_fingerprint(chunks: list[Document]) -> str:
    """给一批 chunk 算一个指纹（内容哈希）。内容一样，指纹必然一样。

    指纹覆盖：chunk 内容 + embedding 模型名 + 维度。
    换模型必然触发重建，防止"旧向量 + 新查询向量"混用的静默 bug。
    """
    hasher = hashlib.sha256()

    # 先哈希模型配置 —— 换模型后即使语料不变，指纹也必然不同
    hasher.update(config.EMBEDDING_MODEL.encode("utf-8"))
    hasher.update(str(config.EMBEDDING_DIM or "default").encode("utf-8"))

    # 再哈希 chunk 内容（按 (source, title_path) 排序，保证顺序稳定）
    ordered = sorted(
        chunks, key=lambda d: (d.metadata["source"], d.metadata["title_path"])
    )
    for doc in ordered:
        hasher.update(doc.metadata["source"].encode("utf-8"))
        hasher.update(doc.metadata["title_path"].encode("utf-8"))
        hasher.update(doc.page_content.encode("utf-8"))
    return hasher.hexdigest()


def _read_saved_fingerprint() -> str | None:
    """读磁盘上存的上一次指纹。索引目录不存在或没存过 → 返回 None。"""
    path = config.INDEX_DIR / FINGERPRINT_FILE
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _write_fingerprint(fingerprint: str) -> None:
    """建完索引后，把这次的指纹存起来，供下次启动比对。"""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / FINGERPRINT_FILE).write_text(fingerprint, encoding="utf-8")


def build_store(chunks: list[Document], fingerprint: str) -> tuple[Chroma, BM25Okapi]:
    """全量重建：把 chunks 向量化 + 建 BM25 索引，都落盘，并记录指纹。"""
    # 1. 向量库
    vector_store = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        persist_directory=str(config.INDEX_DIR),
    )

    # 2. BM25 索引
    bm25 = _build_bm25_index(chunks)
    _save_bm25_index(bm25)

    # 3. 指纹
    _write_fingerprint(fingerprint)

    return vector_store, bm25


def load_existing_store(chunks: list[Document]) -> tuple[Chroma, BM25Okapi]:
    """指纹没变 → 直接加载磁盘上已有的索引（向量库 + BM25），不重新构建。"""
    vector_store = Chroma(
        embedding_function=get_embeddings(),
        persist_directory=str(config.INDEX_DIR),
    )
    bm25 = _load_bm25_index()
    return vector_store, bm25
    #   真正调 API 是 search() 里对用户问题 embed_query 那一刻才发生 ——
    #   加载索引本身完全不联网。


def load_or_build_store(chunks: list[Document]) -> tuple[Chroma, BM25Okapi]:
    """入口函数：指纹一致就读盘，不一致就重建。返回 (向量库, BM25索引)。"""
    fingerprint = compute_fingerprint(chunks)
    if _read_saved_fingerprint() == fingerprint:
        print("[store] 语料指纹未变，直接加载已有索引（向量库 + BM25）")
        return load_existing_store(chunks)

    print("[store] 语料指纹变化（或首次建索引），重新构建中（向量化 + BM25，请稍等）...")
    return build_store(chunks, fingerprint)


def _rrf_fusion(
    results_list: list[list[tuple[Document, float]]], k: int = 60
) -> list[tuple[Document, float]]:
    """RRF（Reciprocal Rank Fusion）融合多路检索结果。

    参数:
        results_list: 多路检索结果，每路是 [(doc, score), ...]
        k: 平滑参数，默认 60（经验值）

    返回:
        融合后的结果，按 RRF 分数降序排列 [(doc, rrf_score), ...]

    原理：
        对每个 doc，RRF 分数 = Σ 1/(k + rank_in_each_list)
        - 在多路都排前面的 doc 会得到更高的 RRF 分数
        - 只在单路出现的 doc 也能保留（分数较低）
    """
    rrf_scores: dict[str, float] = {}  # doc.page_content -> RRF 分数
    doc_map: dict[str, Document] = {}  # doc.page_content -> Document 对象

    for results in results_list:
        for rank, (doc, _score) in enumerate(results, start=1):
            key = doc.page_content  # 用文本内容作为去重 key
            if key not in doc_map:
                doc_map[key] = doc
                rrf_scores[key] = 0.0
            rrf_scores[key] += 1.0 / (k + rank)

    # 按 RRF 分数降序排列
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in sorted_items]


def search(
    vector_store: Chroma,
    bm25: BM25Okapi,
    chunks: list[Document],
    query: str,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """混合检索：向量检索 + BM25 检索 + RRF 融合。

    参数:
        vector_store: Chroma 向量库
        bm25: BM25 索引
        chunks: 所有 chunk（BM25 需要用它定位 doc 对象）
        query: 查询文本
        k: 最终返回多少个结果（默认 config.TOP_K）

    返回:
        [(Document, RRF融合分数), ...] 按分数降序

    流程:
        1. 语义检索：召回 FETCH_K 个（默认 20）
        2. BM25 检索：召回 FETCH_K 个
        3. RRF 融合：根据排名倒数加权
        4. 取前 k 个返回
    """
    final_k = k or config.TOP_K
    fetch_k = config.FETCH_K

    # 1. 语义检索（向量）
    semantic_results = vector_store.similarity_search_with_score(query, k=fetch_k)

    # 2. BM25 检索（关键词）
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    # BM25 返回的是分数数组（和 chunks 顺序一一对应），需要排序取前 fetch_k
    ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:fetch_k]
    bm25_results = [(chunks[i], bm25_scores[i]) for i in ranked_indices]

    # 3. RRF 融合
    fused = _rrf_fusion([semantic_results, bm25_results])

    # 4. 取前 k 个
    return fused[:final_k]