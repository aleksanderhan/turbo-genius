import asyncio
import threading
import time
import websockets
import traceback
import webview
import requests
import json

class ChatApp:
    def __init__(self, server, port):
        self.server = server
        self.port = port
        self.session_id = None

    async def stream_tokens(self, uri, prompt):
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(prompt)
                t0 = time.time()
                num_token = 0
                try:
                    while True:
                        token = await websocket.recv()
                        self.send_to_webview("assistant", token)
                        num_token += 1
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                finally:
                    dt = time.time() - t0
                    print(f"Time elapsed: {dt:.2f} seconds, Number of tokens/sec: {num_token/dt:.2f}, Number of tokens: {num_token}")
        except Exception as e:
            traceback.print_exc()
            self.send_to_webview("system", f"WebSocket connection failed: {e}")

    def send_to_webview(self, role, message):
        sanitized_message = json.dumps(message)
        window.evaluate_js(f'addMessage("{role}", {sanitized_message})')

    def send_message(self, message):
        if self.session_id is None:
            response = requests.get(f"http://{self.server}:{self.port}/session")
            self.session_id = response.json()
            window.evaluate_js(f'addSession("{self.session_id}")')

        prompt = message.strip()
        if prompt:
            try:
                uri = f"ws://{self.server}:{self.port}/stream/{self.session_id}"
                asyncio.run_coroutine_threadsafe(self.stream_tokens(uri, prompt), event_loop)
            except Exception as e:
                traceback.print_exc()
                self.send_to_webview("system", f"Failed to send message: {e}")

    def initialize(self):
        sessions_response = requests.get(f"http://{self.server}:{self.port}/session-list")
        sessions = sessions_response.json()
        for session in sessions:
            session_title = session["title"]
            session_id = session["id"]
            window.evaluate_js(f'addSession("{session_id}")')

    def load_session(self, session_id):
        if session_id != self.session_id:
            session_response = requests.get(f"http://{self.server}:{self.port}/session/{session_id}")
            chat_data = session_response.json()
            print(chat_data)
            for message in chat_data["messages"]:
                if message["role"] == "user" or message["role"] == "assistant":
                    self.send_to_webview(message["role"], message["content"])
            self.session_id = session_id

def start_asyncio_loop():
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store', default="localhost")
    parser.add_argument('--port', action='store', default="8000")
    args = parser.parse_args()

    loop_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
    loop_thread.start()

    app = ChatApp(args.server, args.port)

    window = webview.create_window("Turbo-Genius Chat", "index.html", js_api=app, text_select=True)
    webview.start(app.initialize)
