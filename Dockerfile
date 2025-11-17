FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright setup
RUN playwright install chromium

# Copy application code
COPY . .

# Create directories
RUN mkdir -p sandbox memory_bank agent_states static

# Expose port
EXPOSE 8000

# Environment variables
ENV ENVIRONMENT=production
ENABLE_GCP_TRACING=false
ENABLE_GCP_METRICS=false

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
