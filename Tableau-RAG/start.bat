
@echo off
echo Starting Superstore RAG (Gemini Edition)...
start cmd /k "cd /d D:\Projects\tableau-rag && venv\Scripts\activate && uvicorn api_server:app --port 8000"
timeout /t 2
start cmd /k "cd /d D:\Projects\tableau-rag\extension && python -m http.server 5500"
echo Done! Both services started. Now open Tableau Desktop.
