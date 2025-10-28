import pickle
from pathlib import Path

PIPE_PATH = Path("pipeline_v1.bin")

with PIPE_PATH.open("rb") as f_in:
    pipeline = pickle.load(f_in)

client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
}

proba = pipeline.predict_proba([client])[0, 1]
print(round(float(proba), 3))