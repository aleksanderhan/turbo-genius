# Turbo-Genius
Streaming local llm server and client.

![Turbo-Genius Chat Client](assets/chat_client.gif)


## Features
 - streaming llm server and client
 - session management and persistance
 - automatic title summary
 - syntax highlighting and latex formatting


## Dependencies
    sudo apt-get install sqlite3 libsqlite3-dev

    # for whatever reason these modules have to be installed separately
    pip install packaging torch

    # Install the rest of the dependencies
    pip install -r requirements.txt


## Server
Run any text-generation model from huggingface

### Run in docker
    docker build . --tag="turbo-genius"
    docker run --gpus all -e HUGGINGFACE_TOKEN=<token> -d -p 8000:8000 turbo-genius

### Run in terminal
    python main.py --model <model>


## Clients

### Cli
    python cli.py --host <host> --port <port>

### Chat client desktop app
    python chat_client.py --host <host> --port <port>
