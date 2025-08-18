from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add parent directory to sys.path for module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieve import CompactBERTT5Inference
app = FastAPI()

class Query(BaseModel):
    question: str

# Initialize your QA system once
qa_system = CompactBERTT5Inference(model_dir="/Users/nitish/Documents/Medical Assistant Bot Project /bert_t5_model", kb_csv="/Users/nitish/Documents/Medical Assistant Bot Project /train.csv")

@app.post("/ask")
def ask(query: Query):
    answer = qa_system.answer_question(query.question)
    return {"answer": answer}
