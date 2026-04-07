# Use RunPod's base PyTorch image for maximum compatibility
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler script
COPY handler.py .

# Command to run the handler when the container starts
CMD ["python", "-u", "handler.py"]
