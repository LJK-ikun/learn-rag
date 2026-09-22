# -*- coding: utf-8 -*-
"""
这个文件干什么：把一篇 md（一个 Document）切成很多小块（一堆 Document）。

【为什么不能整篇当一个块】
向量是把一段文字"求平均"的结果。一篇笔记里讲了 HNSW、BM25、pgvector 十几个
话题，压成一个向量就成了"这些话题的平均值"——用户问"BM25 公式是啥"根本匹配
不上。块越小，一个向量表达的意思越纯。

【切法：顺着 Markdown 标题切】
标题不是排版装饰，是作者亲手划的语义边界。作者写 `## 3.1 考勤制度` 的时候，
就是在说"从这里开始讲考勤，到下一个 ## 为止"。这是白送的语义边界，
比按字数硬切强得多。

【切完必须补一步：标题前缀】
按标题切之后，标题被切走当"节名"了，正文里就没有"考勤"这两个字了。
用户问"考勤制度是怎么规定的"，这一块永远搜不到。
所以要把标题路径**拼进正文最前面**，让这块文字自己带着出处：

    【12-补充-检索与RAG.md】补充 A：检索与 RAG > 2.3 索引与向量库

注意是拼进 page_content，**不是塞进 metadata** —— metadata 不参与向量计算，
塞进去等于没写。

【两个必须防的坑，都是实测撞出来的】
① 代码块里的 `# 注释` 长得跟标题一模一样。不拦的话它会被当成 H1，
   把标题栈清空 —— 后面的节全部丢失祖先路径，一直到下一个真 H1。
   实测：551 块里有 176 块（32%）的标题路径是代码注释。
② 标题切完可能还是太长，实测最长 5823 字（上限的 7 倍）。
   一块塞了好几个话题，向量又变回"平均值"了。所以超长的要按段落再切。

这个模块不联网，纯字符串处理，可离线秒跑。
"""
from __future__ import annotations

import re

from langchain_core.documents import Document

import config

# 匹配 Markdown 标题：行首 1~6 个 #，一个空格，然后是标题文字
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# 匹配代码围栏：行首（可带缩进）的 ``` 或 ~~~
FENCE = re.compile(r"^\s*(```|~~~)")


def split_by_heading(text: str) -> list[tuple[list[str], str]]:
    """
    按标题切，返回 [(标题路径, 正文), ...]。

    标题路径是个列表，比如 ["ch12 检索与RAG", "2.3 索引与向量库"]，
    从外到里排列，表示"这一节挂在哪些标题下面"。
    """
    out = []            # 成品：装 (标题路径, 正文) 的列表
    path = []           # 标题栈：我现在读到哪些标题下面
    buf = []            # 正文缓冲：这一节已经攒了哪些行
    in_code = False     # 现在是不是在代码块里

    for line in text.split("\n"):        # 逐行读

        # ---- 代码围栏：翻转状态。这行本身算正文 ----
        if FENCE.match(line):
            in_code = not in_code        # 进代码块 / 出代码块，来回翻
            buf.append(line)
            continue

        # ---- 代码块内部：一律当正文，不判断标题 ----
        # ★ 这是整段代码的关键防线。
        #   笔记里有这种行：  # mewcode/config.py:264-288
        #   它完全符合标题语法（1 个 # + 空格 + 文字），不拦的话会被当成 H1：
        #   后果一：path[:0] 清空标题栈，后面所有节的路径全错
        #   后果二：一节从代码块中间被劈成两半
        if in_code:
            buf.append(line)
            continue

        # ---- 代码块外面，才判断是不是标题 ----
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


def pack(body: str, max_chars: int) -> list[str]:
    """一节太长就切细：按空行（段落）贪心装，装到快满就封口。"""
    if len(body) <= max_chars:           # 绝大多数节走到这里就返回
        return [body]

    pieces = []
    cur = ""
    for para in body.split("\n\n"):      # 按空行切成段落
        # ↑ 为什么按空行切而不是按字数硬切：段落也是语义边界，
        #   在段落中间断开会把一句话劈成两半，两边都读不通。
        if len(cur) + len(para) + 2 <= max_chars:    # +2 是两个换行符的位置
            cur = f"{cur}\n\n{para}" if cur else para
        else:                            # 装不下了
            if cur:
                pieces.append(cur)       # 先把当前这口收掉
            cur = para                   # 这段作为新片的开头
    if cur:
        pieces.append(cur)               # 循环结束还有剩的，别漏

    return pieces
    # 注意：如果某个段落自己就超过 max_chars（长代码块、长表格），
    # 会被原样保留、不再切。代码块从中间断开比超长更糟。


def chunk_document(doc: Document) -> list[Document]:
    """一篇 Document（整篇）→ 一堆 Document（每个带标题前缀）。"""
    source = doc.metadata["source"]      # 文件名，拿来做前缀
    out = []
    for path, body in split_by_heading(doc.page_content):
        prefix = f"【{source}】{' > '.join(path)}" if path else f"【{source}】"
        # ↑ 长这样：【12-补充-检索与RAG.md】补充 A：检索与 RAG > 2.3 索引与向量库
        #   文件名给"哪一篇"，标题路径给"哪一节"，两者信息互补。
        #   代价是多花 ~30 个字，换来的是"这块文字自己能说清自己在讲什么"。

        for piece in pack(body, config.MAX_CHARS):
            out.append(
                Document(
                    page_content=f"{prefix}\n\n{piece}",   # ← 送去算向量的就是这段
                    metadata={
                        "source": source,                 # 来自哪个文件（过滤用）
                        "title_path": " > ".join(path),   # 属于哪一节（引用展示用）
                    },
                )
            )
    return out


def chunk_documents(docs: list[Document]) -> list[Document]:
    """批量切。"""
    out = []
    for d in docs:
        out.extend(chunk_document(d))    # extend 摊平加进来；用 append 会变成嵌套列表
    return out
