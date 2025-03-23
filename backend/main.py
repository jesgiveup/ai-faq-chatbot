from fastapi import FastAPI # handles backend (web server, routing)
from pydantic import BaseModel # handles data validation
from transformers import pipeline # loads pre-trained NLP models from Hugging face

# Create FastAPI instance
app = FastAPI() # this is the backend web application

# Load Hugging Face question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
# "question-answering" is a shortcut so that Hugging Face knows which architecture to load based on it
# "distilbert-base-uncased-distilled-squad" is the actual model name
# the pipeline uses PyTorch so need to download PyTorch
# so I dont have to train anything -> this is just plug and play

# Define the expected format for user input
class QARequest(BaseModel):
    question: str
    context: str

# Root route: when someone visits base URL (/), this function will be returns a message in JSON
@app.get("/") 
def root(): # returns a JSON response
    return {"message": "Hello, I'm your AI FAQ bot!"}

# /ask route: when someone visits /ask, this function will be called
@app.post("/ask")
def ask_qa(payload: QARequest): # takes in a QARequest object
    # payload is the data that the user sends to the server

    # Use the pre-trained model to generate an answer
    result = qa_pipeline({ 
        "question": payload.question,
        "context": payload.context
    })

    # return the model's answer and confidence score
    return {"answer": result["answer"], "score": result["score"]}    


