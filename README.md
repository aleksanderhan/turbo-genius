# Turbo-Genius
Streaming local LLM server and client.

## Features
- streaming LLM server and client
- session management and persistence
- automatic title summary
- syntax highlighting and latex formatting
- GGUF model loading via `llama-cpp-python`

![Turbo-Genius Chat Client](assets/chat_client.gif)

## Dependencies
    sudo apt-get install sqlite3 libsqlite3-dev

    # Install the python dependencies
    pip install -r requirements.txt

## Server
Run a local GGUF chat model with configurable GPU layer offload.

### Run in docker
    docker build . --tag="turbo-genius"
    docker run --gpus all -d -p 8000:8000 -v $(pwd)/models:/app/models turbo-genius

### Run in terminal
    python server.py --model models/model.gguf --n_gpu_layers 20

## Clients

### Cli
    python cli.py --host <host> --port <port>

### Chat client desktop app
    python client.py --host <host> --port <port>
