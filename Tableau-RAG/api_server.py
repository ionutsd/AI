

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_chain import ask

app = FastAPI(title="Superstore RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def answer_question(q: Question):
    try:
        return ask(q.question)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return JSONResponse(content={
                "answer": "⏳ Gemini daily quota exhausted (20 requests/day on free tier). Pandas answers still work! Try: 'sales by region', 'profit by category', 'summary'.",
                "sources_count": 0
            })
        return JSONResponse(content={
            "answer": f"⚠️ Error: {error_msg[:200]}",
            "sources_count": 0
        }, status_code=200)

@app.get("/")
async def health():
    return {"status": "Superstore RAG running! ✦ Powered by Gemini"}
