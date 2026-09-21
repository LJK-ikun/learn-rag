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


# ============================================================
# 直接运行：跑通一次检索
# ============================================================
if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    import chunker
    import loader

    # ① 加载 + 切分 + 拼标题前缀，产出 Document
    #    注意：拼前缀其实是 chunker 的活，chunker.py 目前只产出
    #    (标题路径, 正文)，还没做这步。为了这次演示不卡住先写在这里，
    #    跑通之后要挪回 chunker.py。
    chunks = []
    for doc in loader.load_documents():
        src = doc.metadata["source"]
        for path, body in chunker.split_by_heading(doc.page_content):
            prefix = f"【{src}】{' > '.join(path)}" if path else f"【{src}】"
            chunks.append(
                Document(
                    page_content=f"{prefix}\n\n{body}",
                    metadata={"source": src, "title_path": " > ".join(path)},
                )
            )

    print(f"\n[store] 共 {len(chunks)} 个块，开始向量化（要调 API，请稍等）...")

    # ② 建库
    store = build_store(chunks)

    # ③ 提问，看检索出了什么
    query = "向量库怎么选型"
    print(f"\n提问：{query}\n")
    for i, (doc, score) in enumerate(search(store, query), 1):
        head = doc.page_content.splitlines()[0]
        body = doc.page_content.split("\n\n", 1)[-1].strip().replace("\n", " ")
        print(f"{i}. [{score:.4f}] {head}")
        print(f"   {body[:100]}...\n")
