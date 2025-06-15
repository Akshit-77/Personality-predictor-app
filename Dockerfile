# Use Python 3.11 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Expose Streamlit default port
EXPOSE 8501

# Create a startup script
COPY startup.sh .
RUN chmod +x startup.sh

# Run the startup script
CMD ["./startup.sh"]