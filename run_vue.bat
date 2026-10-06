@echo off
REM Vue 前后端启动脚本
REM 会同时启动后端 API 和前端开发服务器

echo ========================================
echo   RAG 知识库 Web UI (Vue 版)
echo ========================================
echo.

REM 检查依赖是否安装
if not exist ".venv\Scripts\uvicorn.exe" (
    echo [错误] 后端依赖未安装，请先运行:
    echo   .venv\Scripts\pip.exe install -r requirements.txt
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [错误] 前端依赖未安装，请先运行:
    echo   cd frontend
    echo   npm install
    pause
    exit /b 1
)

echo [1/2] 启动后端 API (http://localhost:8000)...
start "RAG-Backend" cmd /k ".venv\Scripts\uvicorn.exe api:app --reload --port 8000"
timeout /t 3 /nobreak >nul

echo [2/2] 启动前端服务 (http://localhost:5173)...
start "RAG-Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   启动完成！
echo ========================================
echo   后端 API: http://localhost:8000
echo   前端界面: http://localhost:5173
echo   API 文档: http://localhost:8000/docs
echo ========================================
echo.
echo 按任意键关闭此窗口 (前后端会继续运行)
pause >nul
