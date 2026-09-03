# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt httpx

# Copy project files
COPY . .

# Pre-calibrate and save initial weights during build if not present
RUN python -c "from core.weights_init import get_calibrated_model; get_calibrated_model('weights/srm_model.pth')"

# Expose port
EXPOSE 8000

# Start Uvicorn server bound to 0.0.0.0
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
