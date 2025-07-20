# Turbo-Genius
Streaming local llm server and client.

## Features
 - streaming llm server and client
 - session management and persistence
 - automatic title summary
 - syntax highlighting and latex formatting
 - image generation capabilities

![Turbo-Genius Chat Client](assets/chat_client.gif)
![Image capabilities](assets/image_capability.png)

## Dependencies
    sudo apt install nvidia-cuda-toolkit
    sudo apt-get install sqlite3 libsqlite3-dev

    # Create a conda env with
    bash 00_create_env.sh

    # for whatever reason these modules have to be installed separately
    pip install packaging torch

    # Install the rest of the dependencies
    pip install -r requirements.txt

## Web App
Run any text-generation model from huggingface

- Run in terminal
    python app.py --model <model> --image_generation --image_cpu_offload
- Open http://localhost:8000 in a browser

