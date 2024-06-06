import asyncio
import threading
import time
import websockets
import traceback
import webview
import requests
import json
import re

class ChatApp:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.session_id = None
        self.session_titles = {}

    async def stream_tokens(self, uri, prompt):
        try:
            async with websockets.connect(uri) as websocket:
                await websocket.send(prompt)
                t0 = time.time()
                num_token = 0
                try:
                    while True:
                        token = await websocket.recv()
                        img_tag_pattern = r'<img\b[^>]*>'
                        if re.search(img_tag_pattern, token):
                            token = token.replace("<host>", self.host).replace("<port>", str(self.port))
                        self.send_to_webview("assistant", token)
                        num_token += 1
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                finally:
                    dt = time.time() - t0
                    print(f"Time elapsed: {dt:.2f} seconds, Number of tokens/sec: {num_token/dt:.2f}, Number of tokens: {num_token}")
                    if self.session_titles[self.session_id] == "New session":
                        self.generate_title(self.session_id)
        except Exception as e:
            traceback.print_exc()
            self.send_to_webview("system", f"WebSocket connection failed: {e}")

    def send_to_webview(self, role, message):
        sanitized_message = json.dumps(message)
        print(sanitized_message)
        window.evaluate_js(f'addMessage("{role}", {sanitized_message})')

    def send_message(self, message):
        if self.session_id is None:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/session")
                self.session_id = str(response.json())
                window.evaluate_js(f'addSession("{self.session_id}", "New session")')
                self.session_titles[self.session_id] = "New session"
            except Exception as e:
                traceback.print_exc()
                self.send_to_webview("system", f"Failed to get session: {e}")

        prompt = message.strip()
        if prompt:
            try:
                uri = f"ws://{self.host}:{self.port}/stream/{self.session_id}"
                asyncio.run_coroutine_threadsafe(self.stream_tokens(uri, prompt), event_loop)
            except Exception as e:
                traceback.print_exc()
                self.send_to_webview("system", f"Failed to send message: {e}")

    def initialize(self):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/session-list")
            sessions = response.json()
            for session in sessions:
                escaped_title = json.dumps(session["title"])
                self.session_titles[str(session["id"])] = escaped_title
                window.evaluate_js(f'addSession("{session["id"]}", {escaped_title})')
        except Exception as e:
            traceback.print_exc()
            self.send_to_webview("system", f"Failed to initialize sessions: {e}")

    def load_session(self, session_id):
        if session_id != self.session_id:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/session/{session_id}")
                chat_data = response.json()
                for message in chat_data["messages"]:
                    if message["role"] == "user" or message["role"] == "assistant":
                        img_tag_pattern = r'<img\b[^>]*>'
                        if re.search(img_tag_pattern, message["content"]):
                            message["content"] = message["content"].replace("<host>", self.host).replace("<port>", str(self.port))
                        self.send_to_webview(message["role"], message["content"])
                self.session_id = session_id
            except Exception as e:
                traceback.print_exc()
                self.send_to_webview("system", f"Failed to load session: {e}")

    def reset_session(self):
        self.session_id = None

    def delete_session(self, session_id):
        try:
            requests.delete(f"http://{self.host}:{self.port}/session/{session_id}")
            if session_id == self.session_id:
                self.reset_session()
            window.evaluate_js(f'clearChat()')
        except Exception as e:
            traceback.print_exc()
            self.send_to_webview("system", f"Failed to delete session: {e}")

    def generate_title(self, session_id):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/session/{session_id}/title")
            title = response.text
            escaped_title = json.dumps(title)
            self.session_titles[session_id] = escaped_title
            window.evaluate_js(f'updateSessionTitle("{session_id}", {escaped_title})')
        except Exception as e:
            traceback.print_exc()
            self.send_to_webview("system", f"Failed to generate title: {e}")

def start_asyncio_loop():
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', action='store', default="localhost")
    parser.add_argument('--port', action='store', default="8000")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    loop_thread = threading.Thread(target=start_asyncio_loop, daemon=True)
    loop_thread.start()

    app = ChatApp(args.host, args.port)

    window = webview.create_window("Turbo-Genius Chat", "index.html", js_api=app, text_select=True)
    webview.start(app.initialize, debug=args.debug)

