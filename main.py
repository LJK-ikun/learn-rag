# -*- coding: utf-8 -*-
"""
这个文件干什么：命令行入口。交互式循环 —— 启动一次，反复问，agent 记得住上下文。

    ./.venv/Scripts/python.exe main.py

输入 exit / quit 退出。

【为什么要改成循环】
"多轮对话记忆"这个能力必须在同一个进程里、连续问多次才能被观察到 ——
之前"传一个参数、跑一次就退出"的模式，每次都是全新进程，agent.py 里的
checkpointer 存的历史活不过一次调用，根本没机会体现"记住了上一轮"。
"""
from __future__ import annotations

import sys
import uuid

import agent as agent_module
import chunker
import loader
from store import load_or_build_store


def main() -> None:
    # 1 建索引（指纹没变就直接读盘，不重新向量化 —— 见 store.load_or_build_store）
    chunks = chunker.chunk_documents(loader.load_documents())
    store = load_or_build_store(chunks)

    # 2 建 agent（带 checkpointer，能记住这个进程里发生过的对话）
    a = agent_module.build_agent(store)

    # 3 这个进程里所有轮次共用同一个 thread_id —— 这就是"记忆"生效的关键。
    #   换一个 thread_id 就等于开了个新会话，跟这边历史互不干扰。
    thread_id = uuid.uuid4().hex
    print(f"\n知识库问答（会话 {thread_id[:8]}，输入 exit/quit 退出）\n")

    while True:
        question = input("你: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = agent_module.ask(a, question, thread_id)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    import io

    # errors="replace" 兜底：万一遇到编不了的字符就打个替身继续，
    # 别让整个程序因为一个怪字符崩掉（loader.py 里读文件时同理）。
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stdin = io.TextIOWrapper(
        sys.stdin.buffer, encoding="utf-8", errors="replace"
    )

    main()
