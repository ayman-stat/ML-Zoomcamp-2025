from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    with Path("pipeline_v2.bin").open("rb") as f:
        pipeline = pickle.load(f)

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(lead: Lead):
    data = lead.model_dump()
    proba = float(pipeline.predict_proba([data])[0, 1])
    return {"subscription_probability": proba}