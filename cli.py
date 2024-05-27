import asyncio
import websockets
from transformers import AutoTokenizer


model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)


messages = [
    {"role": "system", "content": "Help the user with anything they want!"},
    {"role": "user", "content": "Tell me about the connection between the spectral gap problem and the halting problem."},
]


async def stream_tokens(uri, prompt):
    async with websockets.connect(uri) as websocket:
        # Send the prompt to the server
        await websocket.send(prompt)

        # Keep receiving tokens until the connection is closed by the server
        try:
            while True:
                token = await websocket.recv()
                print(token, end='', flush=True)
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed by server")

if __name__ == "__main__":
    # WebSocket server URI including the endpoint
    uri = "ws://localhost:8000/stream"
    # The prompt to send to the model
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=False,
    )

    # Running the async function using asyncio
    asyncio.run(stream_tokens(uri, prompt))
