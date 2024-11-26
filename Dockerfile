FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CLEARML_HOST=http://localhost:8080
ENV MONGODB_URL=mongodb://localhost:27017
ENV QDRANT_URL=localhost:6333

# Run the pipeline and start the application
CMD ["python", "app.py"]