# -*- coding: utf-8 -*-
"""
FastAPI 后端 —— 为前端提供 HTTP 接口

启动方式：
    uvicorn api:app --reload --port 8000

接口列表：
    POST /api/chat       - 发送消息，获取回答
    POST /api/session    - 创建新会话
    GET  /api/stats      - 获取知识库状态
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent as agent_module
import chunker
import loader
from store import load_or_build_store

# ==================== FastAPI 应用 ====================
app = FastAPI(title="RAG Knowledge Base API", version="1.0.0")

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局状态 ====================
class AppState:
    """应用全局状态"""
    agent = None
    chunks = None
    vector_store = None
    bm25 = None
    sessions = {}  # {thread_id: [messages]}

state = AppState()

# ==================== 请求/响应模型 ====================
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[str] = []

class SessionResponse(BaseModel):
    thread_id: str

class StatsResponse(BaseModel):
    chunk_count: int
    session_count: int

# ==================== 启动事件 ====================
@app.on_event("startup")
async def startup_event():
    """应用启动时加载知识库"""
    print("[INFO] 正在加载知识库...")

    # 加载知识库
    chunks = chunker.chunk_documents(loader.load_documents())
    vector_store, bm25 = load_or_build_store(chunks)

    # 构建 agent
    agent = agent_module.build_agent(vector_store, bm25, chunks)

    # 保存到全局状态
    state.agent = agent
    state.chunks = chunks
    state.vector_store = vector_store
    state.bm25 = bm25

    print(f"[INFO] 知识库加载完成（共 {len(chunks)} 个 chunk）")

# ==================== API 接口 ====================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天消息"""
    if state.agent is None:
        raise HTTPException(status_code=503, detail="知识库尚未加载完成")

    # 生成或使用已有的 thread_id
    thread_id = request.thread_id or uuid.uuid4().hex

    # 调用 agent
    try:
        answer = agent_module.ask(state.agent, request.message, thread_id)

        # 解析引用来源（简单实现：从回答文本中提取）
        sources = extract_sources(answer)

        # 保存会话历史
        if thread_id not in state.sessions:
            state.sessions[thread_id] = []
        state.sessions[thread_id].append({
            "role": "user",
            "content": request.message
        })
        state.sessions[thread_id].append({
            "role": "assistant",
            "content": answer
        })

        return ChatResponse(
            answer=answer,
            thread_id=thread_id,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理消息时出错: {str(e)}")

@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    """创建新会话"""
    thread_id = uuid.uuid4().hex
    state.sessions[thread_id] = []
    return SessionResponse(thread_id=thread_id)

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取知识库统计信息"""
    if state.chunks is None:
        raise HTTPException(status_code=503, detail="知识库尚未加载完成")

    return StatsResponse(
        chunk_count=len(state.chunks),
        session_count=len(state.sessions)
    )

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "knowledge_base_loaded": state.agent is not None
    }

# ==================== 工具函数 ====================
def extract_sources(answer: str) -> list[str]:
    """从回答文本中提取引用来源

    示例：
        输入："xxx [来源: 12-补充-检索与RAG.md] yyy"
        输出：["12-补充-检索与RAG.md"]
    """
    import re
    pattern = r'\[来源:\s*([^\]]+)\]'
    matches = re.findall(pattern, answer)
    return [m.strip() for m in matches]

# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
