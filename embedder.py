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
    )