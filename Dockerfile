# --- Step 1: Utilize an official lightweight Python base runtime ---
FROM python:3.11-slim

# --- Step 2: Establish clear operational boundaries inside the container ---
WORKDIR /app

# --- Step 3: Install structural OS dependencies for compilation if needed ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Step 4: Leverage Docker's layer caching for quick package builds ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Step 5: Copy the remaining data pipeline, model objects, and code systems ---
COPY . .

# --- Step 6: Expose the network boundaries (8000 for FastAPI, 8501 for Streamlit) ---
EXPOSE 8000
EXPOSE 8501

# --- Step 7: Define startup behavior (Launches the FastAPI backend engine by default) ---
CMD ["uvicorn", "api.main.py:app", "--host", "0.0.0.0", "--port", "8000"]