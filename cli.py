import asyncio
import websockets
import time
import requests

async def stream_tokens(uri, prompt):
    async with websockets.connect(uri) as websocket:
        # Send the prompt to the server
        await websocket.send(prompt)

        # Keep receiving tokens until the connection is closed by the server
        t0 = time.time()
        num_token = 0
        try:
            while True:
                token = await websocket.recv()
                if token == "[DONE]":
                    break
                print(token, end='', flush=True)
                num_token += 1
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            dt = time.time() - t0
            print("\n")
            print("Time elapsed: {:.2f} seconds".format(dt), "Number of tokens/sec: {:.2f}".format(num_token/dt))
            print()

async def interactive_client():
    # WebSocket server URI including the endpoint
    session_response = requests.get("http://localhost:8000/session")
    uri = f"ws://localhost:8000/stream/{session_response.json()}"

    while True:
        prompt = input(">> ")
        print()
        if prompt.lower() == 'exit':
            break
        await stream_tokens(uri, prompt)

if __name__ == "__main__":
    asyncio.run(interactive_client())
