# -*- coding: utf-8 -*-
"""
Web UI 入口 —— Streamlit 聊天界面

运行方式：
    streamlit run app.py

第一版目标：
- 聊天界面（消息气泡）
- 多轮对话记忆（复用 agent.py 的能力）
- 显示引用来源
"""
from __future__ import annotations

import uuid

import streamlit as st

import agent as agent_module
import chunker
import loader
from store import load_or_build_store


def init_session_state():
    """初始化会话状态（只在第一次访问时执行）"""
    if "initialized" not in st.session_state:
        with st.spinner("🔄 正在加载知识库..."):
            # 1. 加载索引
            chunks = chunker.chunk_documents(loader.load_documents())
            vector_store, bm25 = load_or_build_store(chunks)

            # 2. 构建 agent
            a = agent_module.build_agent(vector_store, bm25, chunks)

            # 3. 存到 session_state（Streamlit 会在同一会话的多次交互间保持）
            st.session_state.agent = a
            st.session_state.chunks = chunks
            st.session_state.initialized = True

    # 会话相关状态
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar():
    """侧边栏：会话管理 + 知识库状态"""
    with st.sidebar:
        st.title("💬 会话管理")

        # 新建会话按钮
        if st.button("➕ 新建会话", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.rerun()

        # 当前会话 ID（缩短显示）
        st.caption(f"当前会话: {st.session_state.thread_id[:8]}")

        st.divider()

        # 知识库状态
        st.title("📚 知识库状态")
        if "chunks" in st.session_state:
            st.metric("Chunk 数量", len(st.session_state.chunks))

        st.divider()
        st.caption("基于混合检索（向量 + BM25）的 RAG 系统")


def render_chat():
    """主聊天区域"""
    st.title("🤖 RAG 知识库问答")

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 用户输入
    if prompt := st.chat_input("问点什么？"):
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 2. 调用 agent
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                answer = agent_module.ask(
                    st.session_state.agent,
                    prompt,
                    st.session_state.thread_id
                )
            st.write(answer)

        # 3. 保存 AI 回复
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


def main():
    # Streamlit 页面配置
    st.set_page_config(
        page_title="RAG 知识库问答",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 初始化
    init_session_state()

    # 渲染界面
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
