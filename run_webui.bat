@echo off
REM Web UI 启动脚本
REM 双击运行即可启动 Streamlit 界面

echo 正在启动 RAG 知识库 Web UI...
echo.
echo 启动后会自动打开浏览器，访问地址: http://localhost:8501
echo 如果没有自动打开，请手动访问上述地址
echo.
echo 按 Ctrl+C 可以停止服务
echo.

.venv\Scripts\streamlit.exe run app.py

pause
