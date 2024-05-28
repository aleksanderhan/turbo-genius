# turbo-genius

Streaming local llm server and client.

## Run in docker
    docker build . --tag="turbo-genius"
    docker run --gpus all -e HUGGINGFACE_TOKEN=<token> -d -p 8000:8000 turbo-genius

## Run in terminal
    pip install -r requirements.txt
    python main.py --model <model>

    python cli.py --server <server> --port <port>