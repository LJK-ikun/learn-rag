# -*- coding: utf-8 -*-
"""
这个文件干什么：命令行入口。问一个问题，打印带出处的答案。

    ./.venv/Scripts/python.exe main.py "向量库怎么选型"
"""
from __future__ import annotations

import sys
import chain
import chunker
import loader
from store import build_store

def main(question: str) -> None:
    # 1 建索引
    chunks = chunker.chunk_documents(loader.load_documents())
    print(f"\n[main] {len(chunks)} 个块， 建索引中（调 API， 请稍等）...")
    store = build_store(chunks)

    # 2 问答
    print(f"\n问题：{question}\n")
    answer, hits =chain.ask(question, store)
    print(answer)

    # 3 出处
    if hits:
        print("\n" + "─" * 60)
        print("引用来源：")
        for i, (doc, score) in enumerate(hits, 1):
            meta = doc.metadata
            print(f"  [{i}] {meta.get('source', '?')}  >  {meta.get('title_path', '')}")
            print(f"      相关度 {score:.4f}")

if __name__ == "__main__":
    import io

    # errors="replace" 兜底：万一遇到编不了的字符就打个替身继续，
    # 别让整个程序因为一个怪字符崩掉（loader.py 里读文件时同理）。
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    q = " ".join(sys.argv[1:]).strip() or "向量库怎么选型"
    main(q)
