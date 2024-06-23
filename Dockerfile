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
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install necessary Python packages first
RUN pip install packaging
RUN pip install vllm
RUN pip install transformers \
                datasets \
                accelerate \
                bitsandbytes \
                flash-attn \
                fastapi \
                uvicorn \ 
                termcolor \
                pygments \
                sqlalchemy \
                diffusers \
                langchain \
                langchain_community \
                langchain-huggingface                
RUN pip install -U "huggingface_hub[cli]"

RUN apt update && apt install 
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Copy the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["tail", "-f", "/dev/null"]