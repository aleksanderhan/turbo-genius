# Use the NVIDIA base image with CUDA
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Upgrade ubuntu
RUN apt-get update && apt-get upgrade -y

# Install Python and necessary system packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install necessary Python packages first
RUN pip install packaging
RUN pip install torch
RUN pip install transformers \
                datasets \
                peft \
                trl \
                accelerate \
                bitsandbytes \
                flash-attn \
                fastapi \
                uvicorn 
RUN pip install -U "huggingface_hub[cli]"


# Copy the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["sh", "-c", "huggingface-cli login --token $HUGGINGFACE_TOKEN && uvicorn main:app --host 0.0.0.0 --port 8000"]

