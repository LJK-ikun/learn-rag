# Web UI 实现：从命令行到可视化界面

## 1. 为什么需要 Web UI

### 痛点
- **命令行门槛高**：非技术用户无法使用
- **不直观**：无法可视化检索过程、引用来源
- **无历史管理**：关掉终端后对话历史丢失
- **难以分享**：无法让团队其他人试用

### 目标
构建一个简洁、易用的 Web 界面，让 RAG 系统的能力能够被更多人使用。

---

## 2. 技术选型：为什么选 Streamlit

### 候选方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Streamlit** | 纯 Python，零前端代码，10 分钟出 MVP | 定制化能力弱，不适合复杂交互 | 内部工具、快速原型 |
| FastAPI + React | 灵活度高，适合生产环境 | 需要前端开发能力，开发周期长 | 正式产品 |
| Gradio | 专为 ML 模型设计，接口简单 | 聊天界面不如 Streamlit 自然 | 模型 demo |
| Flask + Jinja2 | 轻量，完全可控 | 需要手写前端，效率低 | 学习目的 |

**选择 Streamlit 的理由**：
1. **零前端代码**：纯 Python 实现，符合项目技术栈
2. **原生聊天组件**：`st.chat_message` + `st.session_state` 天然适合对话场景
3. **快速迭代**：改代码刷新浏览器即可看到效果
4. **够用**：对于内部工具和学习项目，功能已经足够

---

## 3. 核心实现

### 架构设计

```
app.py (Streamlit)
    │
    ├─ init_session_state()     # 初始化：加载知识库 + 构建 agent
    │   └─ load_or_build_store() ──▶ 复用现有逻辑
    │   └─ build_agent()        ──▶ 复用现有逻辑
    │
    ├─ render_sidebar()         # 侧边栏：会话管理 + 知识库状态
    │   ├─ 新建会话按钮
    │   └─ 显示 chunk 数量
    │
    └─ render_chat()            # 主聊天区
        ├─ 显示历史消息
        └─ 用户输入 ──▶ agent.ask() ──▶ 显示 AI 回答
```

**关键设计决策**：
- **零改动复用**：`app.py` 直接 import `agent.py`、`store.py`，现有代码不需要任何修改
- **状态管理**：利用 `st.session_state` 存储 agent 实例、thread_id、消息历史
- **延迟初始化**：知识库只在第一次访问时加载，后续刷新页面直接复用

### 核心代码解析

#### 1. 初始化（只运行一次）

```python
def init_session_state():
    if "initialized" not in st.session_state:
        with st.spinner("🔄 正在加载知识库..."):
            # 加载知识库（复用现有逻辑）
            chunks = chunker.chunk_documents(loader.load_documents())
            vector_store, bm25 = load_or_build_store(chunks)
            
            # 构建 agent（复用现有逻辑）
            a = agent_module.build_agent(vector_store, bm25, chunks)
            
            # 存到 session_state（Streamlit 会在会话期间保持）
            st.session_state.agent = a
            st.session_state.chunks = chunks
            st.session_state.initialized = True
```

**为什么这样设计**：
- `st.session_state` 是 Streamlit 的会话存储，页面刷新时会保留
- 用 `"initialized"` 标志位确保知识库只加载一次
- 知识库加载耗时 7-54s，必须避免每次交互都重新加载

#### 2. 会话管理

```python
# 会话 ID：每个会话独立的 thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

# 消息历史：存储对话记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 新建会话：生成新 thread_id + 清空历史
if st.button("➕ 新建会话"):
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.rerun()
```

**thread_id 的作用**：
- agent.py 的 `InMemorySaver` 按 thread_id 存储对话历史
- 同一个 thread_id → 多轮记忆生效
- 换一个 thread_id → 开启新对话，与之前的互不干扰

#### 3. 聊天交互

```python
# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("问点什么？"):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. 调用 agent（复用现有逻辑）
    with st.spinner("🤔 思考中..."):
        answer = agent_module.ask(
            st.session_state.agent,
            prompt,
            st.session_state.thread_id
        )
    
    # 3. 保存 AI 回复
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
```

**为什么需要 `st.rerun()`**：
- Streamlit 是脚本式运行：从上到下执行一遍代码生成页面
- 新消息加入后，必须 `rerun()` 重新执行脚本才能显示
- 类似于 React 的 `setState` 触发重新渲染

---

## 4. 使用方式

### 启动步骤

```bash
# 方式 1：命令行启动
.venv\Scripts\streamlit.exe run app.py

# 方式 2：双击脚本启动（Windows）
run_webui.bat
```

启动后会自动打开浏览器，访问 `http://localhost:8501`

### 界面功能

#### 主聊天区
- 输入问题 → Agent 自动检索 → 返回答案 + 引用来源
- 支持多轮对话（上下文记忆）
- 显示"思考中..."加载状态

#### 侧边栏
- **新建会话**：开启新对话，历史清空
- **当前会话 ID**：显示 thread_id 前 8 位（便于 debug）
- **知识库状态**：显示 chunk 总数

---

## 5. 效果展示

### 示例对话

```
用户: HNSW 是什么

AI: HNSW（Hierarchical Navigable Small World）是一种多层图索引，
    用于近似最近邻检索...
    [来源: 12-补充-检索与RAG.md]

用户: 它的参数怎么调

AI: HNSW 主要有三个参数：
    - M：每层的连接数...
    - ef_construction：构建时的搜索宽度...
    - ef_search：查询时的搜索宽度...
    [来源: 12-补充-检索与RAG.md]
    
    ↑ "它"被正确解析为 HNSW（多轮记忆生效）
```

### 与命令行版对比

| 对比项 | 命令行版 | Web UI 版 |
|--------|---------|-----------|
| 启动方式 | `python main.py` | `streamlit run app.py` 或双击 `run_webui.bat` |
| 界面 | 纯文本 | 消息气泡 + 侧边栏 |
| 会话管理 | 无（重启丢失） | 可新建会话、查看当前会话 ID |
| 可视化 | 无 | 知识库状态、加载进度条 |
| 使用门槛 | 需要会用命令行 | 浏览器访问即可 |

---

## 6. 技术细节与坑

### Streamlit 的状态管理机制

**核心概念**：Streamlit 是**脚本式**运行，每次交互都会从头到尾执行一遍 `app.py`。

```python
# 每次交互都会执行这段代码
print("这行代码每次都会打印")

# 只在第一次交互时执行
if "initialized" not in st.session_state:
    print("这行只打印一次")
    st.session_state.initialized = True
```

**常见陷阱**：
1. **忘记用 `st.session_state` 存储状态**
   - 错误：`messages = []` → 每次交互都清空
   - 正确：`st.session_state.messages = []` → 持久化存储

2. **忘记调用 `st.rerun()`**
   - 问题：新消息加入后页面不刷新
   - 解决：在修改 `session_state` 后调用 `st.rerun()`

3. **重复初始化**
   - 问题：每次交互都重新加载知识库（耗时 7-54s）
   - 解决：用标志位 `"initialized"` 确保只初始化一次

### 为什么不修改现有代码

**设计原则**：Web UI 应该是**附加功能**，不应影响现有的命令行版本。

**好处**：
- 命令行用户不受影响
- 降低维护成本（不需要同时维护两套逻辑）
- 代码复用度高（`app.py` 只有 100 行）

**实现方式**：
- `app.py` 作为新的入口，直接 import 现有模块
- 完全不修改 `agent.py`、`store.py`、`tools.py` 等核心模块

---

## 7. 后续优化方向

### 阶段 2：增强引用展示（下一步）
- **问题**：当前引用混在回答文本里，不够突出
- **方案**：
  - 修改 `agent.py` 返回结构化数据（`{"answer": "...", "sources": [...]}`）
  - 在 UI 中单独渲染引用卡片
  - 支持点击展开查看完整 chunk 内容

### 阶段 3：检索可视化
- **功能**：
  - 显示召回的 top-5 chunk
  - 对比向量检索 vs BM25 的排名
  - 展示 RRF 融合权重
- **价值**：帮助理解混合检索的工作原理

### 阶段 4：会话持久化
- **问题**：当前会话存在内存里，刷新页面后丢失
- **方案**：
  - 将会话历史存到 SQLite
  - 侧边栏显示历史会话列表
  - 支持切换到之前的对话继续聊

### 阶段 5：多知识库切换
- **功能**：支持加载不同的笔记目录
- **实现**：侧边栏添加知识库选择器

---

## 8. 总结

### 完成的工作
1. ✅ 创建 `app.py`（100 行代码）
2. ✅ 实现聊天界面（消息气泡 + 输入框）
3. ✅ 会话管理（新建会话、显示会话 ID）
4. ✅ 知识库状态展示（chunk 数量）
5. ✅ 启动脚本 `run_webui.bat`（双击即可运行）
6. ✅ 添加 `streamlit` 依赖到 `requirements.txt`

### 技术亮点
- **零改动复用**：完全不修改现有代码，保持向下兼容
- **快速迭代**：从零到可用界面只用了 100 行代码
- **状态管理**：合理使用 `st.session_state` 避免重复初始化

### 学到的经验
1. **选择合适的工具**：内部工具优先考虑开发效率，Streamlit 是 Python 生态的最佳选择
2. **增量开发**：先做最小可用版本（MVP），验证可行性后再优化
3. **保持兼容**：新功能不应破坏现有功能（命令行版仍然可用）

---

**下一步**：优化引用展示，把来源信息做成可展开的卡片。
