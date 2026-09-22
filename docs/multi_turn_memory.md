# 多轮对话记忆 —— 使用场景、原理、选型对比、延伸八股

> 对应代码：`agent.py`（核心逻辑）、`main.py`（交互式循环）、`tools.py`（工具定义）。
> 关联概念见 `docs/persistence.md`（向量库持久化，同一批语料的另一半故事）。

---

## 一、使用场景与目的

### 1.1 要解决的问题

改造前，`main.py` 是"一次性命令行"模式：

```
./.venv/Scripts/python.exe main.py "HNSW 是什么"
```

每次运行都是一个全新进程，`chain.ask(question, store)` 只看到这一次的问题，
回答完进程就退出。如果紧接着想追问"它的参数怎么调"，模型完全不知道"它"
指的是什么——因为压根没有"上一轮"这个概念留存下来。

这在真实使用场景里很别扭：人问问题很少是一次性的，经常是"先问个大概，
再顺着回答追问细节"。

### 1.2 目的

让同一个会话里的连续提问，模型能"记得"前面问过什么、回答过什么，从而正确
解析"它"“这个”“上面提到的”这类指代，不需要用户每次都把完整上下文重新打一遍。

实测效果（本项目验证）：

```
你: HNSW 是什么
   → 完整介绍 HNSW 原理、参数、和 IVF 的对比

你: 它的参数怎么调
   → 直接列出 M / ef_construction / ef_search 三个参数的调参逻辑，
     没有反问"你说的'它'是什么"
```

### 1.3 适用边界

`InMemorySaver` 这个具体实现只解决"同一个进程内的多轮记忆"——进程一退出，
历史就没了。如果需要"关掉程序、明天再打开，还记得昨天聊了什么"，需要换成
能跨进程持久化的实现（见三、竞对方案）。这和 `docs/persistence.md` 里
"向量索引落盘"是两个独立的持久化问题：一个存的是"语料的向量"，一个存的是
"对话的历史"，互不影响。

---

## 二、底层实现原理

### 2.1 大模型没有记忆，这是一个必须先接受的事实

无论 ChatGPT、DeepSeek 还是任何 LLM API，每次调用在协议层面都是无状态的：
你传一个 `messages` 列表，模型只根据**这一次收到的列表**生成回复，不会主动
去"回忆"之前调用过的内容。所谓"多轮对话记忆"，本质上都是**客户端把历史存起来，
下一轮把"历史 + 新问题"拼在一起再发一遍**：

```
第1轮：messages = [system, user: "HNSW 是什么"]
                                          → assistant: "HNSW 是..."

第2轮：messages = [system, user: "HNSW 是什么", assistant: "HNSW 是...",
                    user: "它的参数怎么调"]
                                          → assistant: "M / ef_construction / ..."
                    ↑ 这一整段是重新发一遍的，不是模型自己记住的
```

这意味着：轮数越多，每次请求实际发送的 token 越多，成本和延迟都会随对话
长度线性增长——这是所有"多轮记忆"方案都绕不开的代价，区别只在于"这部分
拼接、存取的工作谁来做、存在哪"。

### 2.2 checkpointer 做的事：把"存历史、拼历史"自动化

`create_agent` 是 LangChain 在 LangGraph 之上封装的 ReAct 循环（一个
"model 节点 ↔ tools 节点"来回跳的状态图）。`checkpointer` 是挂载在这个图上
的一个组件，职责很单一：**每次 `invoke` 之后把当前的 `messages` 状态存一份，
下次同一个 `thread_id` 再 `invoke` 时，先取出存的状态，再把新消息追加上去**。

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(llm, tools, system_prompt=..., checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "abc123"}}
agent.invoke({"messages": [{"role": "user", "content": "HNSW 是什么"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "它的参数怎么调"}]}, config)
#             ↑ 只传了新问题，但 checkpointer 会自动把上一轮的完整历史拼进去
```

`thread_id` 是"这是哪个会话"的身份证——本项目在 `main.py` 里用
`uuid.uuid4().hex` 给每次启动的进程生成一个，同一进程内所有轮次共用这一个
`thread_id`，这样连续调用才会命中同一份历史；换一个 `thread_id` 就等价于
开一个全新的、互不干扰的会话。

`InMemorySaver` 是 checkpointer 的最简单实现：历史就存在一个进程内的
Python 字典里，`agent.py` 一退出，历史就没了。选它的原因很直接——本项目
当前只需要演示"单进程内的多轮记忆"，`InMemorySaver` 零依赖、不用起任何
额外的存储服务，接口和其他 saver（`SqliteSaver`、`PostgresSaver`）完全一致，
以后要换成跨进程存活的版本，只改 `agent.py` 里 `InMemorySaver()` 这一行。

### 2.3 为什么不手写一个 messages 列表来管历史

最朴素的替代方案是：在 `main.py` 里自己维护一个 `list`，每轮问答完手动
`append` 两条消息（user + assistant），下一轮把整个列表传给模型。这个方案
完全可行，甚至更容易看清"历史到底是怎么被拼接的"这个底层机制。

本项目最终选了 `checkpointer` 而不是手写列表，原因是：

- **和 `create_agent` 的架构是一体的**：本项目的 agent 本身就是
  `create_agent` 构建的 ReAct 循环（见 `agent.py`），`checkpointer` 是这个
  框架原生支持的能力，接进去只是加一个构造参数；手写列表则需要绕开
  `create_agent` 的状态管理，自己接管 `invoke` 的输入输出，等于在框架之外
  另起一套逻辑。
- **以后换存储介质的成本更低**：`InMemorySaver → SqliteSaver` 只改一行；
  手写列表如果以后想让历史跨进程存活，要么自己写文件读写，要么手动接入
  数据库，等于重新发明 checkpointer 已经做好的事。

代价是：`checkpointer` 内部具体怎么存取、怎么拼接历史，对使用者是不透明的
（黑盒）——如果只是为了理解"多轮记忆的本质是重新发送历史"这个概念，手写
一次列表拼接反而更直观。这是"用框架能力换透明度"的典型取舍。

### 2.4 agent.py 的实际结构

```python
def build_agent(store):
    sources = [f.name for f in loader.find_markdown_files()]
    tools = build_tools(store, sources)
    return create_agent(
        get_llm(),
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

def ask(agent, question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config
    )
    return result["messages"][-1].content
```

`main.py` 只建一次 agent（循环外），循环内反复 `ask(agent, question, thread_id)`，
`thread_id` 在整个进程生命周期内固定不变——这是让 checkpointer 在多轮之间
真正"连上"的关键，如果每轮都生成新的 `thread_id`，效果就退化回"每轮都是
全新对话"，和没有 checkpointer 没有区别。

---

## 三、竞对方案及优缺点对比

### 3.1 记忆存储介质对比

| 方案                             | 存储介质                 | 跨进程存活 | 部署成本                       | 适合规模         | 备注                                                             |
| -------------------------------- | ------------------------ | ---------- | ------------------------------ | ---------------- | ------------------------------------------------------------------ |
| **手写 messages 列表**（未采用） | 进程内存（Python list）  | ❌         | 零依赖                         | 学习/演示        | 最透明，但要自己管理拼接、截断、持久化，工程量随需求增长          |
| **InMemorySaver**（本次选型）    | 进程内存（dict）         | ❌         | 零依赖                         | 单进程会话       | 和手写列表的存储介质其实一样，区别是"存取拼接"的逻辑交给框架管   |
| **SqliteSaver**                  | 本地 sqlite 文件         | ✅         | 单机嵌入式，约等于零额外部署  | 单机、多次启动   | 关机重启后历史还在，适合"一个人用的本地工具"场景                 |
| **PostgresSaver**                | Postgres 数据库          | ✅         | 依赖一个 Postgres 实例         | 多用户、生产环境 | 能和业务数据同库同事务，支持多用户会话隔离、并发写入             |
| **Redis 系 saver**（社区方案）   | Redis                    | ✅         | 依赖一个 Redis 实例            | 高并发、短会话   | 适合"历史只需要保留几小时/几天"的场景（配 TTL 自动过期）         |

### 3.2 为什么本项目选 InMemorySaver

- **当前只需要验证"多轮记忆"这个能力本身**：跑一次进程、连续问几轮、验证
  "它"能被正确解析——这个目标下，历史是否跨进程存活并不重要。
- **换成别的 saver 成本极低**：接口是 LangGraph 的 `BaseCheckpointSaver`
  统一抽象，`agent.py` 里只有 `InMemorySaver()` 这一行会变，`build_agent`
  的其余逻辑、`ask` 函数完全不用动——这和 `docs/persistence.md` 里
  "`VectorStore` 抽象让换 Chroma 只改一行"是同一种设计收益。
- **不想引入额外的运维成本**：本项目当前是单人单机使用，起一个 sqlite 文件
  或 Postgres 实例来存对话历史，收益和这份额外复杂度不成正比。

如果这个项目以后要做成"能被多人使用的服务"，或者需要"关掉再打开还记得
之前聊过什么"，应该换成 `SqliteSaver`（单机场景）或 `PostgresSaver`
（多用户场景）——这和之前向量库从 `InMemoryVectorStore` 换成 `Chroma`
是完全类似的升级路径：接口不变，只换持久化介质。

### 3.3 判断标准（通用结论）

> "记忆只需要活过一次进程运行" → 内存存储就够（`InMemorySaver` / 手写列表）。
> "记忆需要活过进程重启" → 必须落盘（sqlite / Postgres / Redis 等）。
> 选哪个持久化介质，取决于"单机还是多机"“单用户还是多用户”这两个问题，
> 和向量库选型的判断逻辑是同一套思路。

---

## 四、延伸技术栈八股（面试高频）

### Q1：LLM 的"多轮对话记忆"在协议层面到底是怎么回事？

LLM API 本身是无状态的——每次请求只根据这次传入的 `messages` 生成回复，
不存在"服务端帮你记住上一次说了什么"这种机制（除非模型服务商自己在
背后做了会话缓存，但那是供应商私有实现，不是协议保证）。"多轮记忆"
永远是客户端职责：存历史、下一轮把"历史 + 新输入"整体重发。这也直接
决定了一个副作用——**对话越长，每次请求的 token 数越多，成本和延迟
都线性增长**，这是所有多轮对话系统都要面对的问题，解决办法通常是
"历史摘要"（把老消息压缩成摘要而不是全量保留）或"滑动窗口"（只保留
最近 N 轮）。本项目当前没有实现这两种优化，属于已知局限（见五）。

### Q2：checkpointer 和"手写一个消息列表"相比，解决了什么额外问题？

看起来两者存储的都是"一份 messages 历史"，但 checkpointer 额外处理了：

- **多会话隔离**：`thread_id` 天然支持"同一个 agent 实例同时服务多个
  互不干扰的会话"，手写列表如果要支持多会话，得自己维护
  `dict[thread_id, list]`，本质上是在重新实现 checkpointer 的一部分。
- **和状态图的其他状态一起管理**：`create_agent` 底层是 LangGraph 的
  `StateGraph`，除了 `messages`，还可能有其他状态字段（比如工具调用的
  中间结果）；checkpointer 是对**整个图状态**做快照，不是只存消息列表，
  这一点手写方案很难覆盖全。
- **存储介质可插拔**：见三、竞对方案，接口一致、换底层实现成本低。

### Q3：为什么 thread_id 要在整个进程/会话生命周期内保持不变？

`thread_id` 是 checkpointer 用来"从存储里取出对应历史"的唯一键。如果每次
调用都生成新的 `thread_id`，checkpointer 每次都会查到"这是个全新会话，
没有历史"，效果等价于完全没接 checkpointer——多轮记忆能力名存实亡。
反过来，如果两个本该独立的会话用了同一个 `thread_id`，会导致历史串台
（用户 A 的问题看到用户 B 的上下文），这是多用户场景下必须做会话隔离的
原因（通常用用户 ID、session ID 拼出唯一的 `thread_id`）。

### Q4：为什么"历史越长，成本越高"，实际系统怎么控制这个问题？

因为每一轮请求都要把全部历史重新发送一遍（协议无状态，见 Q1），历史消息
数和 token 数是单调递增的。常见工程手段：

- **滑动窗口**：只保留最近 N 轮，超出的直接丢弃——简单，但会丢失早期
  上下文（比如用户很早提到的一个约束，后面还想引用）。
- **摘要压缩**：定期把"老消息"用 LLM 压缩成一段摘要，替换掉原始消息——
  保留了信息密度，但压缩本身要多调一次模型，且摘要可能丢细节。
- **检索式记忆**：把历史消息也做成向量存起来，每轮只检索"和当前问题最
  相关的历史片段"拼进上下文，而不是无脑全量重发——这其实是把"记忆问题"
  转化成了本项目已经在做的"检索问题"（见 `docs/persistence.md`），
  是长对话场景下更现代的方案。

本项目当前规模（几轮问答、单会话）没有触发这个问题，属于"当前不需要，
但要知道存在"的延伸知识点。

### Q5：本项目的 agent 架构（`create_agent` + ReAct 循环）和最早的固定
`chain.py` 方案比，多轮记忆这件事有什么本质区别？

`chain.py` 是"检索 → 拼提示词 → 调模型"的固定线性流程，没有"决策"这一步——
每次都无条件检索、无条件把结果塞进提示词。`agent.py` 用 `create_agent`
构建的是一个"model 节点 ↔ tools 节点"的循环图，模型自己决定要不要调用
工具、调用几次、什么时候停。多轮记忆在这两种架构下都能加（`chain.py`
一样可以手动拼历史），但 `create_agent` 天然提供了 `checkpointer` 这个
挂载点，把"记住历史"这件事和"决策要不要调用工具"这件事都统一在同一个
状态图的生命周期里管理，工程上更省心；代价是要理解 LangGraph 状态图这
一层抽象，比线性 chain 多一层认知负担。

---

## 五、当前方案的已知局限（留给后续迭代）

1. **历史不持久化**：`InMemorySaver` 存在进程内存里，进程一退出，之前的
   对话历史全部丢失，下次启动是全新会话。如果需要"关掉程序、明天继续
   接着聊"，需要换成 `SqliteSaver` 或 `PostgresSaver`（见三、竞对方案）。
2. **没有历史长度控制**：当前实现是"全量保留、全量重发"，没有做滑动窗口
   或摘要压缩（见 Q4）。对话轮数一多，每轮请求的 token 数和延迟会持续
   增长，本项目目前的使用规模（几轮问答）还感觉不到这个问题，但这是一个
   已知的、需要在对话变长后重新评估的缺口。
3. **单进程假设**：`thread_id` 由 `main.py` 在进程启动时用 `uuid4` 生成，
   没有做"跨进程复用同一个 thread_id"或"多用户会话隔离"的设计，这两点
   在本项目当前"单人单机交互式使用"的场景下不是问题，但如果要做成服务
   给多人用，需要重新设计 `thread_id` 的生成和传递方式（比如从外部请求里
   带上用户/会话标识）。
