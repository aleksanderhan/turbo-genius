# Turbo-Genius
Streaming local llm server and client.

![Turbo-Genius Chat Client](assets/chat_client.gif)

## Server
Run any text-generation model from huggingface

### Run in docker
    docker build . --tag="turbo-genius"
    docker run --gpus all -e HUGGINGFACE_TOKEN=<token> -d -p 8000:8000 turbo-genius

### Run in terminal
    pip install -r requirements.txt
    python main.py --model <model>


## Clients

### Cli
    python cli.py --server <server> --port <port>

### Chat client desktop app
    python chat_client.py --server <server> --port <port>
