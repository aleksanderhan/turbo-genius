# Turbo-Genius

Streaming local llm server and client.

## Server

### Run in docker (not working atm)
    docker build . --tag="turbo-genius"
    docker run --gpus all -e HUGGINGFACE_TOKEN=<token> -d -p 8000:8000 turbo-genius

### Run in terminal
    pip install -r requirements.txt
    python main.py --model <model>


## Clients

### Cli
    python cli.py --server <server> --port <port>

### Tkinter desktop app
    python chat.py --server <server> --port <port>

