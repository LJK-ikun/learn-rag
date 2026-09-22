# -*- coding: utf-8 -*-
"""
这个文件干什么：把 `config.INTERVIEW_DIR` 下的 .md 读成 LangChain 的 `Document` 对象。

【为什么需要它】
后面四步（切分 → 向量化 → 入库 → 检索）需要一个统一的数据结构来传递。
不定这个结构的话，每个模块都得自己 open 文件、自己拼路径 —— 改一处要改四处。

【Document 的两块内容，分工是整个 RAG 里最容易被忽略的一点】

    Document(
        page_content="HNSW 建了一个多层图……",           # 参与向量计算
        metadata={"source": "12-补充-检索与RAG.md",      # 不参与计算
                  "title_path": "2.3 索引与向量库"}
    )

- page_content：会被 embedding 压成向量，**只有它参与相似度计算**
- metadata：不参与计算，用途只有两个 —— ①检索前的过滤 ②答案里的引用展示

记住这条：**metadata 里写的字，检索时是"看不见"的。**
想让一段信息能被搜到，必须写进 page_content。

【为什么不用 LangChain 的 DocumentLoader】
langchain_community 没装（为这点功能装一个几百 MB 的包不值得），
而读 20 个本地 md 本来也就是 glob + read_text 两行的事。

这个模块**不联网**，可离线秒跑。
"""
from __future__ import annotations          # 让类型注解"延后求值"，于是能写 list[Path] 这种新语法
                                            # 又不用管运行时的 Python 版本 —— 反正只是给人和工具看的

from pathlib import Path                    # Path：面向对象的路径。比 os.path 拼字符串干净得多

from langchain_core.documents import Document   # LangChain 全链路流转的那个数据结构，就它一个

import config                               # 我们自己的配置中心：语料在哪、切多大都问它


def find_markdown_files(directory: Path | None = None) -> list[Path]:
    """找出语料目录下所有 .md 文件，按文件名排序（保证每次跑顺序一致）。"""
    directory = directory or config.INTERVIEW_DIR
    # ↑ 参数给了就用参数的，没给就用配置里的。
    #   为什么留这个口子：测试时想指向一个只有 2 个文件的小目录，不用改 .env。

    if not directory.exists():              # 目录不存在 → 后面 glob 会静静返回空列表
                                            # 那种"读了 0 个文件"的错最难查，所以这里主动拦一下
        raise SystemExit(                   # SystemExit 而不是 Exception：这是"环境没配好"，
                                            # 不是程序 bug，不需要堆栈，给一句人话就行
            f"\n[找不到语料] {directory} 不存在。\n"
            f"  改 config 里的 INTERVIEW_DIR，或在 .env 里设 INTERVIEW_DIR。\n"
        )

    files = sorted(directory.glob("*.md"))
    # ↑ glob("*.md") 只找这一层，不递归子目录（`rglob` 才递归）。
    #   sorted() 很重要：不排序的话，glob 返回的顺序取决于文件系统内部实现，
    #   会导致每次建索引时 chunk 的编号都不一样 —— 调试时你会以为代码有问题。

    files = [f for f in files if f.name.upper() != "README.MD"]
    # ↑ 列表推导式过滤掉 README。`.upper()` 是为了兼容 readme.md / README.MD 各种大小写。
    #   为什么排除它：README 是导航页，正文基本是一堆链接，当语料纯属噪声。

    return files                            # 返回 list[Path]，交给下面的 load_one 逐个读


def load_one(path: Path) -> Document:
    """读一个 md 文件，返回一个 Document（整个文件先当一个 Document，切分交给 chunker）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    # ↑ encoding 必须显式写 utf-8：Windows 默认是 GBK，读中文 md 会直接 UnicodeDecodeError。
    #   errors="replace" 兜底：万一个别字符编不了就打个替身继续，
    #   别因为一个奇怪字符让整个建索引过程崩掉（50 万字的语料里出现一个怪字符太正常了）。

    return Document(                        # 造一个 Document 对象并直接返回，不写中间变量
        page_content=text,                  # 全文原样塞进去。注意这里**不切** —— 切分是 chunker 的事
                                            # 单一职责：loader 只管"读进来"，不管"怎么切"
        metadata={                          # metadata 不参与向量计算，只用于过滤和展示
            "source": path.name,            # 文件名（不含目录）。后面做元数据过滤和引用展示都用它
            "path": str(path),              # 完整路径。转成 str 是因为 metadata 最终要序列化成 JSON，
        },                                  # 而 Path 对象不能直接序列化 —— 现在转掉，省得以后报错
    )


def load_documents(directory: Path | None = None) -> list[Document]:
    """读全部语料，返回 list[Document]。"""
    files = find_markdown_files(directory)  # 第一步：找到所有文件路径
    docs = [load_one(f) for f in files]     # 第二步：列表推导式，逐个读成 Document
    print(f"[loader] 从 {directory or config.INTERVIEW_DIR} 读了 {len(docs)} 个文件")
    # ↑ 打一行日志。为什么用 print 而不是 logging：这是个学习项目，
    #   print 的输出直接可见，不用配 handler。真要上生产再换。
    return docs                             # 交给调用方（chunker / store）
