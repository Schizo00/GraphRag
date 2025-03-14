from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from main import call
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class History(BaseModel):
    question: str = None
    answer: str = None

class Message(BaseModel):
    question: str = None
    history: List[History] = None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/ping")
def root():
    return {
        "ping" : "pong"
    }

@app.post("/chat")
def chat(message: Message):
    return call(message.question, message.history)