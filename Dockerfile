FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# We use opencv-python-headless, so we don't need libgl1-mesa-glx
# We only need libglib2.0-0 for some core dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
