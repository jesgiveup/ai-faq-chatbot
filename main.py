from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

print("✅ Starts loading Hugging Face model.")

# Load Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

print("✅ Hugging Face model loaded and ready.")

# Store context globally (simple memory store)
context_holder = {"text": ""}

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": "",
        "question": "",
        "context": context_holder["text"]
    })

@app.post("/ask", response_class=HTMLResponse)
def ask_question(
    request: Request,
    context: str = Form(...),
    question: str = Form(...)
):
    # Update context if provided
    if context.strip():
        context_holder["text"] = context

    # Use existing context
    result = qa_pipeline({
        "question": question,
        "context": context_holder["text"]
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": result["answer"],
        "question": question,
        "context": context_holder["text"]
    })
