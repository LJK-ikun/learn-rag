# -*- coding: utf-8 -*-
"""
这个文件干什么：agent —— 让模型自己决定要不要调用工具，并且**记住对话历史**。

    第1轮 问题 ──▶ create_agent(模型, [search_notes, list_notes], checkpointer=...)
                         │
                         ├─ 存进 checkpointer（按 thread_id 分组）
                         ▼
    第2轮 问题 ──▶ 同一个 thread_id ──▶ agent 自动把第1轮的历史也带上再问模型

【记忆的本质】
大模型本身没有记忆——每次调用 API，模型只看到这一次传过去的 messages 列表。
"多轮对话记忆"就是把历史存起来，下一轮把"历史 + 新问题"一起再发一遍。
checkpointer 就是干这件事的组件：它按 thread_id（会话 ID）存一份 messages 历史，
你只需要在每次 invoke 时告诉它"这是哪个会话"，取历史、拼新消息、存回去这一套
流程就都在它内部自动完成了。

这里用的是 InMemorySaver：历史存在进程内存里，进程一退出就没了。
换成 SqliteSaver / PostgresSaver 能让历史跨进程存活，接口不变，以后要换很容易。
"""
from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

import loader
from chain import get_llm
from tools import build_tools

SYSTEM_PROMPT = """你是一个基于私人笔记回答问题的助手。

你有两个工具：
- search_notes：检索笔记里的具体内容，回答技术问题时用它。
- list_notes：列出笔记文件清单，回答"你有哪些笔记"这类问题时用它。

严格遵守：
1. 只依据工具返回的资料回答，不要使用资料之外的任何知识。
2. 每一条结论后面标出处（工具返回的内容前会带文件名）。
3. 工具查不到相关内容就直接说「笔记里没有相关内容」，不要编造。
4. 用中文回答，简洁、直接，不要客套话。"""


def build_agent(vector_store, bm25, chunks):
    """建一个带记忆的 agent：塞好模型 + 工具 + 系统提示 + checkpointer。"""
    sources = [f.name for f in loader.find_markdown_files()]
    tools = build_tools(vector_store, bm25, chunks, sources)
    return create_agent(
        get_llm(),
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def ask(agent, question: str, thread_id: str) -> str:
    """跑一轮：把问题交给 agent，拿回最终答案文本。

    thread_id 是"这是哪个会话"的身份证——同一个 thread_id 连续调用，
    历史会自动累加；换一个 thread_id 就是全新对话，跟之前的互不干扰。
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config
    )
    return result["messages"][-1].content
