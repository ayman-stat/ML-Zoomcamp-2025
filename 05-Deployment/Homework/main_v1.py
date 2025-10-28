from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

# load once at startup
pipeline = None
pipe_path = Path("pipeline_v1.bin")

@app.on_event("startup")
def load_pipeline():
    global pipeline
    with pipe_path.open("rb") as f_in:
        pipeline = pickle.load(f_in)

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/predict")
def predict(lead: Lead):
    features = lead.model_dump()
    proba = float(pipeline.predict_proba([features])[0, 1])
    return {"subscription_probability": proba}