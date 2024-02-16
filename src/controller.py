import uvicorn

from fastapi import FastAPI
from chain import invoke_ai, ChatRequest, ChatResponse
from ingest import ingest_docs, IngestResponse

app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
def chat_interact(chat_request: ChatRequest):
    return invoke_ai(chat_request)


@app.post("/ingest", response_model=IngestResponse)
def load_docs():
    return ingest_docs()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)