# RAG 知识库问答系统

基于混合检索（向量 + BM25 + RRF 融合）的个人笔记问答系统，支持多轮对话记忆和 Agent 自主决策。

## 项目简介

这是一个用于个人面试笔记的 RAG（检索增强生成）问答系统。通过混合检索技术和 Agent 架构，能够准确回答技术问题并标注引用来源，支持上下文追问。

**核心价值：**
- 解决"笔记越来越多，找不到想要的知识点"的痛点
- 通过混合检索提升召回率（覆盖语义理解 + 精确关键词匹配）
- Agent 架构让系统能自主决策何时检索、如何组织答案

## 核心特性

### 1. 混合检索架构
- **向量检索**：基于语义理解，能匹配"同义改写"（如"图结构快不快" → "HNSW 查询性能"）
- **BM25 关键词检索**：精确匹配术语（如用户问"BM25"时不会漏掉关键文档）
- **RRF 融合**：两路检索结果按排名倒数加权融合，在两路都排前面的文档得分更高

### 2. Agent 驱动 + 多轮记忆
- **ReAct 循环**：模型自主决策何时调用 `search_notes` 或 `list_notes` 工具
- **多轮记忆**：基于 LangGraph InMemorySaver，支持上下文追问（"HNSW 是什么" → "它的参数怎么调"）
- **引用标注**：每条答案自动标注来源文件，可追溯

### 3. 持久化优化
- **指纹机制**：基于内容哈希判断语料是否变化，避免重复向量化
- **效果**：首次构建 ~54s，缓存命中 ~7s（提升 87%）
- **Chroma 向量库**：本地 sqlite 持久化，重启后直接加载

## 技术栈

| 模块       | 技术选型                                    | 说明                                   |
| ---------- | ------------------------------------------- | -------------------------------------- |
| 向量化     | Alibaba DashScope `text-embedding-v4`       | 1024 维，中文友好                      |
| 对话模型   | DeepSeek `deepseek-v4-flash`                | 性价比高，推理速度快                   |
| 向量库     | Chroma                                      | 本地 sqlite，单机嵌入式部署            |
| 关键词检索 | BM25 (rank-bm25)                            | 经典 TF-IDF 扩展，适合中文分词         |
| Agent 框架 | LangChain + LangGraph                       | ReAct 循环 + 多轮记忆                  |
| 语料切分   | 自定义 Markdown 切分器                      | 按标题层级切分 + 代码块保护 + 段落二次切分 |

**依赖清单**：见 `requirements.txt`（约 50MB，主要是 chromadb）

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd rag-learning

# 创建虚拟环境（Python 3.10+）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：
```ini
# 向量化（阿里 DashScope）
DASHSCOPE_API_KEY=sk-xxxxx

# 对话（DeepSeek）
DEEPSEEK_API_KEY=sk-xxxxx

# 语料目录（指向你的笔记目录）
INTERVIEW_DIR=D:\Project\my_code\interview
```

### 3. 运行

```bash
# 首次运行会自动构建索引（约 1 分钟）
.venv/Scripts/python.exe main.py  # Windows
# python main.py  # Linux/Mac

# 启动后进入交互式对话
你: HNSW 是什么
[Agent 自动调用 search_notes 工具检索...]
AI: HNSW 是层次化可导航小世界图...

你: 它的参数怎么调
[自动记住上文的"它"指 HNSW...]
AI: 主要有三个参数：M（每层连接数）、ef_construction...

# 输入 exit 或 quit 退出
```

## 项目结构

```
rag-learning/
├── main.py              # 入口：交互式循环
├── agent.py             # Agent 构建：ReAct + 多轮记忆
├── tools.py             # 工具定义：search_notes / list_notes
├── chain.py             # 早期固定 chain 方案（已被 agent 替代）
├── store.py             # 向量库：构建/加载/检索/混合检索/指纹机制
├── embedder.py          # 向量化封装
├── loader.py            # 文档加载（读 .md 文件）
├── chunker.py           # 自定义切分器（按标题层级切分）
├── config.py            # 配置中心（所有参数集中管理）
├── requirements.txt     # 依赖清单
├── .env.example         # 环境变量模板
├── .gitignore
├── index/               # 向量索引落盘目录（自动生成，已 ignore）
└── docs/                # 技术文档
    ├── persistence.md          # 持久化机制详解
    ├── multi_turn_memory.md    # 多轮对话记忆详解
    └── 05_混合检索.md           # 混合检索架构详解
```

### 核心模块说明

| 文件          | 职责                                                         | 核心函数/类                           |
| ------------- | ------------------------------------------------------------ | ------------------------------------- |
| `config.py`   | 配置中心：API Key、模型名、切分参数、检索参数                | `EMBEDDING_MODEL`, `CHAT_MODEL`, ...  |
| `loader.py`   | 加载笔记：递归读取 `INTERVIEW_DIR` 下的 .md 文件             | `load_documents()`                    |
| `chunker.py`  | 切分策略：按 Markdown 标题切分 + 代码块保护 + 段落二次切分   | `chunk_documents()`                   |
| `embedder.py` | 向量化封装：包装 DashScope embedding API                     | `get_embeddings()`                    |
| `store.py`    | 向量库核心：构建/加载/检索/指纹机制/混合检索                 | `load_or_build_store()`, `search()`   |
| `tools.py`    | Agent 工具：`search_notes`（检索）+ `list_notes`（列清单）   | `build_tools()`                       |
| `agent.py`    | Agent 构建：ReAct 循环 + InMemorySaver 多轮记忆              | `build_agent()`, `ask()`              |
| `main.py`     | 交互式入口：while 循环 + 同一 thread_id 保持会话连续性       | `main()`                              |

## 技术亮点

### 1. 混合检索 + RRF 融合

**问题背景**：
- 纯向量检索：擅长语义，但对精确术语不敏感（问"BM25"可能召回一堆"检索算法"但漏掉真正提到 BM25 的段落）
- 纯关键词检索：擅长精确匹配，但不懂同义词改写

**解决方案**：
```python
# 两阶段检索
semantic_results = vector_store.similarity_search(query, k=20)  # 语义召回 20 个
bm25_results = bm25.get_scores(query_tokens)[:20]               # 关键词召回 20 个
fused = rrf_fusion([semantic_results, bm25_results])            # RRF 融合
return fused[:5]                                                # 取前 5 个
```

**RRF（Reciprocal Rank Fusion）**：
- 对每个文档，RRF 分数 = Σ 1/(k + rank_in_each_list)
- 在两路都排前面的文档 → RRF 分数高
- 只在单路出现的文档 → 也保留但分数低

### 2. 指纹机制避免重复向量化

**问题**：843 个 chunk 全量向量化耗时 ~54s，但 90% 的情况下语料并未改变。

**解决方案**：
```python
def compute_fingerprint(chunks):
    hasher = hashlib.sha256()
    # 哈希模型配置（换模型必须重建）
    hasher.update(config.EMBEDDING_MODEL.encode())
    hasher.update(str(config.EMBEDDING_DIM).encode())
    # 哈希所有 chunk 内容
    for doc in sorted(chunks):
        hasher.update(doc.page_content.encode())
    return hasher.hexdigest()

# 启动时对比指纹
if saved_fingerprint == current_fingerprint:
    load_from_disk()  # 7s
else:
    rebuild()         # 54s
```

### 3. Agent 架构 + 多轮记忆

**Agent 决策示例**：
```
用户问："你有哪些笔记"
  → Agent 选择调用 list_notes()（不需要检索）

用户问："HNSW 的 M 参数是什么"
  → Agent 选择调用 search_notes(query="HNSW M 参数")
  → 从检索结果中提取答案 + 标注来源
```

**多轮记忆**：
```python
# agent.py
checkpointer = InMemorySaver()  # 按 thread_id 存历史
agent = create_agent(llm, tools, checkpointer=checkpointer)

# main.py
thread_id = uuid4().hex  # 整个进程共用同一个 thread_id
while True:
    answer = ask(agent, question, thread_id)  # 历史自动累加
```

## 效果展示

### 示例对话 1：多轮记忆

```
你: HNSW 是什么

AI: HNSW（Hierarchical Navigable Small World）是一种多层图索引，
    用于近似最近邻检索。它通过构建层次化的图结构，在查询时从顶层
    快速定位大致区域，再逐层下探到精确位置。
    [来源: 12-补充-检索与RAG.md]

你: 它的参数怎么调

AI: HNSW 主要有三个参数：
    - M：每层的连接数，常见取值 16~64，越大召回率越高但内存越大
    - ef_construction：构建时的搜索宽度，影响建索引速度和质量
    - ef_search：查询时的搜索宽度，越大召回越好但速度越慢
    [来源: 12-补充-检索与RAG.md]
    ↑ 注意这里的"它"被正确解析为 HNSW
```

### 示例对话 2：混合检索

```
你: BM25 算法的核心思想

AI: BM25 是基于 TF-IDF 的改进算法，核心思想：
    1. 词频饱和：多出现几次的词不会无限制提升分数（用 k1 参数控制）
    2. 文档长度归一化：长文档不会仅因为长就得高分（用 b 参数控制）
    公式：score = IDF(qi) * (f(qi) * (k1+1)) / (f(qi) + k1 * (1-b + b*|D|/avgdl))
    [来源: 12-补充-检索与RAG.md]
    
    ↑ 这种问题因为精确提到"BM25"这个术语，关键词检索会起关键作用
```

## 已知局限与改进空间

### 当前局限

1. **历史不持久化**：`InMemorySaver` 存在进程内存，重启后对话历史丢失
   - 改进方向：换成 `SqliteSaver` 或 `PostgresSaver`

2. **无历史长度控制**：对话轮数多了会导致每次请求 token 数线性增长
   - 改进方向：滑动窗口 / 历史摘要压缩

3. **增量更新缺失**：改一个字也要全量重建索引
   - 当前规模（843 块）重建 1 分钟，可接受
   - 规模大了需要增量更新策略

4. **缺少评估体系**：没有量化指标衡量检索质量
   - 改进方向：引入 RAGAS 或自定义评估集

### 后续规划

- [ ] 效果评估：准备测试集，对比混合检索 vs 单一检索的召回率
- [ ] Web UI：用 Streamlit 做可视化界面
- [ ] 单元测试：覆盖 chunker、store、agent 核心逻辑
- [ ] Docker 化：一键部署
- [ ] 检索可视化：展示 RRF 融合前后的排序变化

## 开发日志

### 已完成里程碑

- **M1-M3**: 基础模块（loader / chunker / embedder / store）
- **M4**: chain 架构（检索 → 拼提示词 → DeepSeek → 带引用的答案）
- **M5a**: 持久化（InMemoryVectorStore → Chroma + 指纹机制）
- **M5b**: 混合检索（向量 + BM25 + RRF 融合）
- **M5c**: Agent 架构（ReAct 循环 + 多轮记忆）

详细技术文档见 `docs/` 目录。

## 常见问题

**Q: 为什么向量化和对话用两家不同的服务？**  
A: DeepSeek 没有 embedding 接口，只有 chat 接口。embedding 和 chat 本来就是两个独立能力，各挑最优即可。

**Q: 为什么不用 LlamaIndex / LangChain 的内置 RAG pipeline？**  
A: 为了学习目的，手写了完整的检索链路（loader → chunker → embedder → store → agent），能更清楚理解每个环节的细节和权衡。

**Q: 843 个 chunk 是什么规模？**  
A: 对应 20 个 Markdown 文件（我的面试笔记），约 20 万字。切分策略是按 Markdown 标题层级切分，每块最多 800 字。

**Q: 能处理多大规模的语料？**  
A: Chroma + HNSW 索引理论上能撑到几十万到百万级 chunk。当前规模（千级）完全不需要考虑性能优化。

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提 Issue 或 PR。
