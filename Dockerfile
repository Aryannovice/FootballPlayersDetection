# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flask

# Copy the application code
COPY . .

# Expose port for health check
EXPOSE 8080

# Create a simple health check endpoint
RUN echo 'from flask import Flask, jsonify\n\
import os\n\
\n\
app = Flask(__name__)\n\
\n\
@app.route("/health")\n\
def health_check():\n\
    return jsonify({\n\
        "status": "healthy",\n\
        "service": "player-tracking",\n\
        "version": "1.0.0",\n\
        "approaches": ["brute_force", "bytetrack_reid", "homography_matching"]\n\
    })\n\
\n\
@app.route("/")\n\
def root():\n\
    return jsonify({\n\
        "message": "Player Tracking Service",\n\
        "endpoints": ["/health"],\n\
        "approaches": {\n\
            "approach1": "brute_force and brute_force_reid",\n\
            "approach2": "bytetrack_reid and homography_matching"\n\
        }\n\
    })\n\
\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=8080)' > health_server.py

# Default command to run the health server
CMD ["python", "health_server.py"]
