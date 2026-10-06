# Vue 前端项目说明

## 技术栈
- Vue 3 + TypeScript
- Vite (构建工具)
- 原生 CSS (无 UI 框架)

## 项目结构
```
frontend/
├── src/
│   ├── App.vue          # 根组件
│   ├── Chat.vue         # 聊天界面主组件
│   ├── api.ts           # API 请求封装
│   └── main.ts          # 入口文件
├── index.html
├── package.json
└── vite.config.ts
```

## 开发命令

```bash
# 安装依赖
npm install

# 启动开发服务器 (http://localhost:5173)
npm run dev

# 构建生产版本
npm run build

# 预览生产构建
npm run preview
```

## API 接口

前端会调用后端的以下接口：
- `POST /api/chat` - 发送消息
- `POST /api/session` - 创建新会话
- `GET /api/stats` - 获取统计信息
- `GET /api/health` - 健康检查

**注意**：确保后端 API 在 `http://localhost:8000` 运行。

## 快速启动

**方式 1：一键启动（推荐）**
```bash
# 在项目根目录双击
run_vue.bat
```

**方式 2：手动启动**
```bash
# 终端 1：启动后端
.venv\Scripts\uvicorn.exe api:app --reload --port 8000

# 终端 2：启动前端
cd frontend
npm run dev
```

然后在浏览器访问 http://localhost:5173
