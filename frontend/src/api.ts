// API 请求封装
const API_BASE_URL = 'http://localhost:8000/api'

interface ChatRequest {
  message: string
  thread_id?: string
}

interface ChatResponse {
  answer: string
  thread_id: string
  sources: string[]
}

interface SessionResponse {
  thread_id: string
}

interface StatsResponse {
  chunk_count: number
  session_count: number
}

// 发送聊天消息
export async function chatAPI(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  return await response.json()
}

// 创建新会话
export async function createSessionAPI(): Promise<SessionResponse> {
  const response = await fetch(`${API_BASE_URL}/session`, {
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  return await response.json()
}

// 获取统计信息
export async function getStatsAPI(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE_URL}/stats`)

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  return await response.json()
}

// 健康检查
export async function healthCheckAPI(): Promise<{ status: string; knowledge_base_loaded: boolean }> {
  const response = await fetch(`${API_BASE_URL}/health`)

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }

  return await response.json()
}
