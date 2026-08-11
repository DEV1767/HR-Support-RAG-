from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from src.model import rag_chain

app = FastAPI(
    title="NovaTech Solutions Pvt. Ltd.",
    description="AI based HR Support RAG API",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "RAG server is running"
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = rag_chain.invoke(
        {
            "question": request.question
        }
    )

    return {
        "answer": answer
    }


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):

    def generate():

        for chunk in rag_chain.stream(
            {
                "question": request.question
            }
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )