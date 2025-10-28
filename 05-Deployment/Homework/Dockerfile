FROM agrigorev/zoomcamp-model:2025

# Copy only what we need
WORKDIR /code

# Copy dependency files (if present)
COPY pyproject.toml uv.lock* /code/

# Install uv
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir uv

# Instead of 'uv sync', install dependencies directly (avoids src/ structure issue)
RUN uv pip install --system fastapi uvicorn scikit-learn==1.6.1 numpy scipy pydantic requests click h11 starlette anyio

# Copy your FastAPI app
COPY main_v2.py /code/main_v2.py

EXPOSE 8000

# Run the API
CMD ["uvicorn", "main_v2:app", "--host", "0.0.0.0", "--port", "8000"]