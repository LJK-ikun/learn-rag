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

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from embedder import get_embeddings


FINGERPRINT_FILE = "fingerprint.txt"


def compute_fingerprint(chunks: list[Document]) -> str:
    """给一批 chunk 算一个指纹（内容哈希）。内容一样，指纹必然一样。"""
    hasher = hashlib.sha256()
    # 先按 (source, title_path) 排序，保证哈希结果不受列表顺序影响 ——
    # 万一以后 loader/chunker 换成并发处理，产出顺序就不一定稳定了。
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


def build_store(chunks: list[Document], fingerprint: str) -> Chroma:
    """全量重建：把 chunks 向量化，落盘到 INDEX_DIR，并记录这次的指纹。"""
    store = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        persist_directory=str(config.INDEX_DIR),
    )
    # ↑ 多传的 persist_directory 就是持久化的关键：告诉 Chroma 把数据落盘
    #   到这个目录下的 sqlite 文件里，不传就只存内存，等于白装了持久化。
    _write_fingerprint(fingerprint)
    return store


def load_existing_store() -> Chroma:
    """指纹没变 → 直接加载磁盘上已有的索引，不重新向量化。"""
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=str(config.INDEX_DIR),
    )
    # ↑ 这里的 get_embeddings() 只是"记住用哪个模型"（供以后 search 用），
    #   真正调 API 是 search() 里对用户问题 embed_query 那一刻才发生 ——
    #   加载索引本身完全不联网。


def load_or_build_store(chunks: list[Document]) -> Chroma:
    """入口函数：指纹一致就读盘，不一致就重建。main.py 只需要调这一个函数。"""
    fingerprint = compute_fingerprint(chunks)
    if _read_saved_fingerprint() == fingerprint:
        print("[store] 语料指纹未变，直接加载已有索引（不重新向量化）")
        return load_existing_store()

    print("[store] 语料指纹变化（或首次建索引），重新向量化中（调 API，请稍等）...")
    return build_store(chunks, fingerprint)


def search(
    store: Chroma, query: str, k: int | None = None
) -> list[tuple[Document, float]]:
    """检索，返回 [(Document, 分数), ...]。

    注意分数含义：Chroma 返回的是**距离**，分数越小越相似 —— 跟以前
    InMemoryVectorStore 返回余弦相似度（越大越像）正好相反。以后如果
    有代码按"分数越大越像"做排序或展示，必须跟着反过来。
    """
    return store.similarity_search_with_score(query, k=k or config.TOP_K)
    # 用 with_score 而不是 similarity_search：后者不带分数。
    # 调检索效果时分数是唯一的反馈信号，看不见分数就只能瞎猜。