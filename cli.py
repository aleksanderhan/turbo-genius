import asyncio
import websockets

async def stream_tokens(uri, prompt):
    async with websockets.connect(uri) as websocket:
        # Send the prompt to the server
        await websocket.send(prompt)

        # Keep receiving tokens until the connection is closed by the server
        try:
            while True:
                token = await websocket.recv()
                print(token, end='')
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed by server")

if __name__ == "__main__":
    # WebSocket server URI including the endpoint
    uri = "ws://192.168.1.13:8000/stream"
    # The prompt to send to the model
    prompt = "Tell me about the connection between the spectral gap problem and the halting problem."

    # Running the async function using asyncio
    asyncio.run(stream_tokens(uri, prompt))
