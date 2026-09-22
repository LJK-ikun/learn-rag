# -*- coding: utf-8 -*-
"""
这个文件干什么：RAG 的最后一步 —— 把检索到的资料 + 问题交给大模型，拿回带引用的答案。

    问题 ──▶ 检索(store.py) ──▶ 5 块资料 ──▶ ★拼提示词★ ──▶ DeepSeek ──▶ 带引用的答案
                                              ↑ 就是这里

提示词是这里唯一"用自然语言写业务规则"的地方。
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

import config
from store import search

SYSTEM_PROMPT = """你是一个基于私人笔记回答问题的助手。

严格遵守以下规则：

1. **只依据下面提供的「资料」回答**，不要使用资料之外的任何知识。
2. 每一条结论后面必须标出处编号，例如 [1] 或 [2][3]。
3. 如果资料里没有足够的信息回答这个问题，就直接说「笔记里没有相关内容」，
   并简要说明资料里实际讲的是什么。**不要猜测，不要编造，不要用常识补齐。**
4. 用中文回答，简洁、直接，不要客套话。"""

    
def get_llm() -> ChatDeepSeek:
 """造一个对话模型客户端。"""
 return ChatDeepSeek(
    model=config.CHAT_MODEL,
    api_key=config.require("DEEPSEEK_API_KEY"),   # 到这一刻才检查 key
    base_url=config.CHAT_BASE_URL,                # 别名，字段名其实是 api_base
    temperature=0.1,
)

def build_context(hits: list[tuple[Document, float]]) -> str:
    """把检索结果拼成带编号的资料块。编号就是引用的锚点。"""
    block = []
    for i, (doc, _score) in enumerate(hits, start=1):
        # ↑ enumerate(..., 1): 从 1 开始编号，不是从0
        # 模型和人类都习惯[1] 开头，[0] 看着别扭
        block.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(block)


def ask(
      question: str, store, k: int | None = None
) -> tuple[str, list[tuple[Document, float]]]:
   """一次完整的问答：检索 → 拼提示词 → 调模型。"""
   hits = search(store, question, k=k)  # 检索，拿回 [(Document, 分数), ...]

   if not hits:
      return "（检索不到任何资料）", []
   
   context = build_context(hits)  # 拼提示词里的资料块

   messages = [
      SystemMessage(content=SYSTEM_PROMPT),
      HumanMessage(
         content=f"资料：\n{context}\n\n问题：{question}\n\n请根据资料回答问题，并严格遵守规则。"
      ),
   ]

   reply = get_llm().invoke(messages)  # 调模型，拿回答案
   return reply.content, hits