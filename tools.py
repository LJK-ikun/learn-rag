# -*- coding: utf-8 -*-
"""
这个文件干什么：把知识库的检索能力，包装成 agent 能调用的「工具」。

【工具到底是什么】
一个工具 = 一个普通函数 + 一份**给模型看的说明书**（docstring）+ 参数说明。

关键：模型看不到你的代码，它只看到函数名、docstring、参数表。
      所以 docstring 不是"给人看的注释"，是"给模型看的 API 文档"。
      它写得清不清楚，直接决定模型会不会用、用得对不对。

【为什么用工厂函数而不是全局变量】
工具需要 store 才能检索，而 store 是运行时才建出来的。
要么塞全局变量（脏、没法测），要么把 store 当参数传进来（干净）。
我们选后者：build_tools(store) 返回一组已经"记住"了 store 的工具。
"""
from __future__ import annotations

from langchain_core.tools import tool

import config
from store import search


def build_tools(vector_store, bm25, chunks, sources: list[str]) -> list:
    """造出这个 agent 能用的全部工具。"""

    @tool
    def search_notes(query: str) -> str:
        """在面试笔记里检索相关片段。

        什么时候该用：问题涉及具体的技术概念、实现细节、面试八股，
        需要引用笔记里的原话来回答。
        什么时候不该用：闲聊、寒暄，或者问的是"你有哪些笔记"这类
        关于笔记范围本身的问题（那种情况用 list_notes）。

        Args:
            query: 检索用的查询词。可以直接用用户的问题原话，
                也可以改写成更容易命中的说法 —— 比如用户问
                "那个图结构的东西快不快"，改写成 "HNSW 查询性能 复杂度"。
        """
        hits = search(vector_store, bm25, chunks, query, k=config.TOP_K)
        if not hits:
            return "没有检索到相关内容。"
        return "\n\n".join(doc.page_content for doc, _ in hits)
    
    @tool
    def list_notes() -> str:
        """列出知识库里都有哪些笔记文件。

        什么时候该用：用户问的是"笔记的范围"而不是"笔记的内容" ——
        比如"你手头有哪些资料""有没有讲 XX 主题的笔记""一共几篇"。
        什么时候不该用：用户问的是某个具体技术问题，那是 search_notes 的活。

        没有参数：它不需要查什么，只是把清单原样端出来。
        """
        if not sources:
            return "知识库里没有任何笔记。"
        return "\n".join(f"- {s}" for s in sources)

    return [search_notes, list_notes]