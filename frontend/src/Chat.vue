<!-- 聊天界面主组件 -->
<template>
  <div class="chat-container">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <h2>💬 会话管理</h2>
        <button @click="createNewSession" class="btn-new-session">
          ➕ 新建会话
        </button>
      </div>

      <div class="session-info">
        <p class="session-id">当前会话: {{ shortThreadId }}</p>
      </div>

      <div class="divider"></div>

      <div class="stats">
        <h3>📚 知识库状态</h3>
        <div class="stat-item">
          <span class="stat-label">Chunk 数量</span>
          <span class="stat-value">{{ stats.chunkCount }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">会话数量</span>
          <span class="stat-value">{{ stats.sessionCount }}</span>
        </div>
      </div>

      <div class="footer">
        <p>基于混合检索的 RAG 系统</p>
      </div>
    </aside>

    <!-- 主聊天区 -->
    <main class="chat-main">
      <div class="chat-header">
        <h1>🤖 RAG 知识库问答</h1>
      </div>

      <!-- 消息列表 -->
      <div class="messages" ref="messagesContainer">
        <div
          v-for="(msg, index) in messages"
          :key="index"
          :class="['message', msg.role]"
        >
          <div class="message-content">
            <div class="message-text" v-html="formatMessage(msg.content)"></div>

            <!-- 引用来源（仅 AI 回答） -->
            <div v-if="msg.role === 'assistant' && msg.sources && msg.sources.length > 0" class="sources">
              <div class="sources-header">📎 引用来源</div>
              <div class="source-list">
                <span
                  v-for="(source, idx) in msg.sources"
                  :key="idx"
                  class="source-tag"
                >
                  {{ source }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- 加载中提示 -->
        <div v-if="isLoading" class="message assistant">
          <div class="message-content">
            <div class="loading">
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
              <span class="loading-text">思考中...</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入框 -->
      <div class="input-area">
        <input
          v-model="inputMessage"
          @keypress.enter="sendMessage"
          :disabled="isLoading"
          placeholder="问点什么？"
          class="input-box"
        />
        <button
          @click="sendMessage"
          :disabled="isLoading || !inputMessage.trim()"
          class="btn-send"
        >
          发送
        </button>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { chatAPI, createSessionAPI, getStatsAPI } from './api'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
}

interface Stats {
  chunkCount: number
  sessionCount: number
}

// 状态
const messages = ref<Message[]>([])
const inputMessage = ref('')
const isLoading = ref(false)
const threadId = ref<string>('')
const stats = ref<Stats>({ chunkCount: 0, sessionCount: 0 })
const messagesContainer = ref<HTMLElement | null>(null)

// 计算属性
const shortThreadId = computed(() => {
  return threadId.value ? threadId.value.slice(0, 8) : '加载中...'
})

// 初始化
onMounted(async () => {
  await createNewSession()
  await loadStats()
})

// 创建新会话
async function createNewSession() {
  try {
    const response = await createSessionAPI()
    threadId.value = response.thread_id
    messages.value = []
  } catch (error) {
    console.error('创建会话失败:', error)
    alert('创建会话失败，请检查后端是否启动')
  }
}

// 发送消息
async function sendMessage() {
  const content = inputMessage.value.trim()
  if (!content || isLoading.value) return

  // 添加用户消息
  messages.value.push({
    role: 'user',
    content
  })

  // 清空输入框
  inputMessage.value = ''
  isLoading.value = true

  // 滚动到底部
  await nextTick()
  scrollToBottom()

  try {
    // 调用 API
    const response = await chatAPI({
      message: content,
      thread_id: threadId.value
    })

    // 添加 AI 回复
    messages.value.push({
      role: 'assistant',
      content: response.answer,
      sources: response.sources
    })

    // 更新统计
    await loadStats()
  } catch (error) {
    console.error('发送消息失败:', error)
    messages.value.push({
      role: 'assistant',
      content: '抱歉，处理消息时出错了。请确保后端服务正常运行。'
    })
  } finally {
    isLoading.value = false
    await nextTick()
    scrollToBottom()
  }
}

// 加载统计信息
async function loadStats() {
  try {
    const data = await getStatsAPI()
    stats.value = {
      chunkCount: data.chunk_count,
      sessionCount: data.session_count
    }
  } catch (error) {
    console.error('加载统计信息失败:', error)
  }
}

// 格式化消息（保留换行）
function formatMessage(text: string) {
  return text.replace(/\n/g, '<br>')
}

// 滚动到底部
function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  background: #f5f5f5;
}

/* ==================== 侧边栏 ==================== */
.sidebar {
  width: 280px;
  background: white;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.sidebar-header h2 {
  font-size: 18px;
  margin-bottom: 16px;
  color: #333;
}

.btn-new-session {
  width: 100%;
  padding: 10px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}

.btn-new-session:hover {
  background: #0056b3;
}

.session-info {
  margin-top: 16px;
}

.session-id {
  font-size: 12px;
  color: #666;
  font-family: monospace;
}

.divider {
  height: 1px;
  background: #e0e0e0;
  margin: 20px 0;
}

.stats h3 {
  font-size: 16px;
  margin-bottom: 12px;
  color: #333;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  font-size: 14px;
}

.stat-label {
  color: #666;
}

.stat-value {
  color: #007bff;
  font-weight: bold;
}

.footer {
  margin-top: auto;
  padding-top: 20px;
  font-size: 12px;
  color: #999;
  text-align: center;
}

/* ==================== 主聊天区 ==================== */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
}

.chat-header {
  padding: 20px 24px;
  border-bottom: 1px solid #e0e0e0;
  background: white;
}

.chat-header h1 {
  font-size: 24px;
  color: #333;
  margin: 0;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
}

.message {
  margin-bottom: 20px;
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message.assistant {
  justify-content: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 8px;
  line-height: 1.6;
}

.message.user .message-content {
  background: #007bff;
  color: white;
}

.message.assistant .message-content {
  background: #f0f0f0;
  color: #333;
}

.message-text {
  font-size: 15px;
}

/* 引用来源 */
.sources {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #ddd;
}

.sources-header {
  font-size: 12px;
  color: #666;
  margin-bottom: 8px;
}

.source-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.source-tag {
  display: inline-block;
  padding: 4px 10px;
  background: white;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
  color: #555;
}

/* 加载动画 */
.loading {
  display: flex;
  align-items: center;
  gap: 8px;
}

.loading-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #666;
  animation: loading 1.4s infinite ease-in-out both;
}

.loading-dot:nth-child(1) {
  animation-delay: -0.32s;
}

.loading-dot:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes loading {
  0%, 80%, 100% {
    transform: scale(0);
  }
  40% {
    transform: scale(1);
  }
}

.loading-text {
  font-size: 14px;
  color: #666;
}

/* ==================== 输入区 ==================== */
.input-area {
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  border-top: 1px solid #e0e0e0;
  background: white;
}

.input-box {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 15px;
  outline: none;
  transition: border-color 0.2s;
}

.input-box:focus {
  border-color: #007bff;
}

.input-box:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.btn-send {
  padding: 12px 24px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 15px;
  transition: background 0.2s;
}

.btn-send:hover:not(:disabled) {
  background: #0056b3;
}

.btn-send:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
