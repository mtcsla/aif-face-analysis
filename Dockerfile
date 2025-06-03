# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY resources ./resources

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
