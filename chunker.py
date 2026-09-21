# -*- coding: utf-8 -*-
"""
这个文件干什么：把一篇 md（一个 Document）切成很多小块。

【为什么不能整篇当一个块】
向量是把一段文字"求平均"的结果。一篇笔记里讲了 HNSW、BM25、pgvector 十几个
话题，压成一个向量就成了"这些话题的平均值"——用户问"BM25 公式是啥"根本匹配
不上。块越小，一个向量表达的意思越纯。

【切法：顺着 Markdown 标题切】
标题不是排版装饰，是作者亲手划的语义边界。作者写 `## 3.1 考勤制度` 的时候，
就是在说"从这里开始讲考勤，到下一个 ## 为止"。这是白送的语义边界，
比按字数硬切强得多。

这一版只做**结构切分**这一件事，别的先都不加。

这个模块不联网，纯字符串处理，可离线秒跑。
"""
from __future__ import annotations

import re

# 匹配 Markdown 标题：行首 1~6 个 #，一个空格，然后是标题文字
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_by_heading(text: str) -> list[tuple[list[str], str]]:
    """
    按标题切，返回 [(标题路径, 正文), ...]。

    标题路径是个列表，比如 ["ch12 检索与RAG", "2.3 索引与向量库"]，
    从外到里排列，表示"这一节挂在哪些标题下面"。
    """
    out = []            # 成品：装 (标题路径, 正文) 的列表
    path = []           # 标题栈：我现在读到哪些标题下面
    buf = []            # 正文缓冲：这一节已经攒了哪些行

    for line in text.split("\n"):        # 逐行读
        m = HEADING.match(line)
        if m:                            # 是标题
            _flush(out, path, buf)       # 先把上一节收口存起来
            level = len(m.group(1))      # "##" 长度是 2，所以 level = 2
            path = path[: level - 1] + [m.group(2)]
            # ↑ "退栈再压入"：读到几级标题，就保留到它的上一级
            #   level=1 → path[:0] 全砍光，从根重来
            #   level=2 → path[:1] 只留 H1，砍掉旧的 H2/H3
            #   level=3 → path[:2] 留 H1+H2，砍掉旧的 H3
            #   效果：遇到同级或更高级的标题时，已经结束的旧路径自动被丢掉
            continue

        buf.append(line)                 # 不是标题 → 正文，攒起来

    _flush(out, path, buf)               # 最后一段没有"下一个标题"来收口，手动收
    return out


def _flush(out: list, path: list, buf: list) -> None:
    """把攒着的正文收口成一节。全空白就不要，避免产出空块。"""
    body = "\n".join(buf).strip()        # 把行拼成字符串，去掉首尾空白
    if body:                             # 有内容才收
        out.append((list(path), body))   # list(path) 复制一份：列表存的是引用，
                                         # 直接存 path 的话，后面 path 一变这里也跟着变
    buf.clear()                          # 清空缓冲，准备攒下一节


# ============================================================
# 直接运行：看看切分结果
# ============================================================
if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    import loader

    docs = loader.load_documents()

    total = 0
    for d in docs:
        total += len(split_by_heading(d.page_content))
    print(f"\n{len(docs)} 篇 → 共 {total} 个分节")

    # 先盯着一篇看：标题路径对不对、有没有被劈开
    doc = docs[11]
    print(f"\n=== {doc.metadata['source']}（前 25 个分节）===\n")
    for path, body in split_by_heading(doc.page_content)[:25]:
        indent = "  " * (len(path) - 1) if path else ""
        name = path[-1] if path else "(无标题正文)"
        print(f"{indent}{name}   [{len(body)} 字]")
