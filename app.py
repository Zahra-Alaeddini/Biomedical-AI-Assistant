from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import webbrowser
import threading
import time

from rag_agent import HebbianNeuroGraphBrain

app = FastAPI(title="Medical Chat - Professional UI")

app.mount("/static", StaticFiles(directory="static"), name="static")

agent = HebbianNeuroGraphBrain()

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1> index.html not found </h1><p> Be sure file exists in the static directory! </p>"

@app.post("/chat")
async def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer = agent.answer_query(request.query.strip())
        return {"role": "assistant", "content": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def open_browser():
    time.sleep(0.05)
    webbrowser.open_new("http://localhost:8000")

if __name__ == "__main__":
    print("Server is running...")
    print("Browser will be open automaticaly: http://localhost:8000")
    print("If does not open, use this link manually! \n")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)