# turbo-genius

    docker build . --tag="turbo-genius"
    docker run --gpus all -e HUGGINGFACE_TOKEN=<token> -d -p 8000:8000 turbo-genius