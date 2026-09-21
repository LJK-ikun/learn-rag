# 向量化工具

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

import config

def get_embeddings() -> OpenAIEmbeddings:
    """造一个配置好的 embedding 客户端。"""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.require("DASHSCOPE_API_KEY"),   # ← 到这一刻才检查 key
        base_url=config.EMBEDDING_BASE_URL,
        dimensions=config.EMBEDDING_DIM,               # None = 用模型默认 1024
        check_embedding_ctx_length=False,              # 发送原始文本，dashscope的做法
        chunk_size=10,                                 # 一批发多少条，见下
    )
    # ↑ chunk_size：LangChain 默认 1000（照 OpenAI 定的），
    #   但 DashScope 一批最多 20 条，超了直接 400 报错。
    #   这里取 10 留一半余量 —— 上限值各家不一样，余量比压满更省心。