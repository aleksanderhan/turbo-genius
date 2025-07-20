from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
import asyncio
import json
import re
import requests
import traceback
from typing import Dict, List

app = FastAPI()

# Store active WebSocket connections
connections: Dict[str, WebSocket] = {}

class ChatApp:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.session_titles = {}

    async def stream_tokens(self, uri, prompt, websocket: WebSocket):
        try:
            import websockets
            async with websockets.connect(uri) as backend_ws:
                await backend_ws.send(prompt)
                try:
                    while True:
                        token = await backend_ws.recv()
                        img_tag_pattern = r'<img\b[^>]*>'
                        if re.search(img_tag_pattern, token):
                            token = token.replace("<host>", self.host).replace("<port>", str(self.port))
                        
                        # Send to frontend via WebSocket
                        await websocket.send_json({
                            "type": "message",
                            "role": "assistant",
                            "content": token
                        })
                except websockets.exceptions.ConnectionClosed:
                    print("Backend connection closed")
        except Exception as e:
            traceback.print_exc()
            await websocket.send_json({
                "type": "error",
                "message": f"Backend connection failed: {e}"
            })

    def get_session_list(self):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/session-list")
            return response.json()
        except Exception as e:
            traceback.print_exc()
            return []

    def create_session(self):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/session")
            return str(response.json())
        except Exception as e:
            traceback.print_exc()
            return None

    def load_session(self, session_id):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/session/{session_id}")
            chat_data = response.json()
            
            # Process image tags
            for message in chat_data["messages"]:
                if message["role"] in ["user", "assistant"]:
                    img_tag_pattern = r'<img\b[^>]*>'
                    if re.search(img_tag_pattern, message["content"]):
                        message["content"] = message["content"].replace("<host>", self.host).replace("<port>", str(self.port))
            
            return chat_data
        except Exception as e:
            traceback.print_exc()
            return None

    def delete_session(self, session_id):
        try:
            requests.delete(f"http://{self.host}:{self.port}/session/{session_id}")
            return True
        except Exception as e:
            traceback.print_exc()
            return False

    def generate_title(self, session_id):
        try:
            response = requests.get(f"http://{self.host}:{self.port}/session/{session_id}/title")
            return response.json()
        except Exception as e:
            traceback.print_exc()
            return None

# Initialize chat app (you'll need to pass host/port as environment variables or config)
chat_app = ChatApp("localhost", "8000")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = id(websocket)
    connections[connection_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "send_message":
                session_id = data.get("session_id")
                message = data.get("message")
                
                if not session_id:
                    session_id = chat_app.create_session()
                    if session_id:
                        await websocket.send_json({
                            "type": "session_created",
                            "session_id": session_id
                        })
                
                if session_id and message:
                    uri = f"ws://{chat_app.host}:{chat_app.port}/stream/{session_id}"
                    await chat_app.stream_tokens(uri, message, websocket)
                    
            elif action == "load_session":
                session_id = data.get("session_id")
                chat_data = chat_app.load_session(session_id)
                if chat_data:
                    await websocket.send_json({
                        "type": "session_loaded",
                        "data": chat_data
                    })
                    
            elif action == "delete_session":
                session_id = data.get("session_id")
                success = chat_app.delete_session(session_id)
                await websocket.send_json({
                    "type": "session_deleted",
                    "success": success,
                    "session_id": session_id
                })
                
            elif action == "get_sessions":
                sessions = chat_app.get_session_list()
                await websocket.send_json({
                    "type": "sessions_list",
                    "sessions": sessions
                })
                
            elif action == "generate_title":
                session_id = data.get("session_id")
                title = chat_app.generate_title(session_id)
                if title:
                    await websocket.send_json({
                        "type": "session_title_updated",
                        "session_id": session_id,
                        "title": title
                    })
                
    except WebSocketDisconnect:
        if connection_id in connections:
            del connections[connection_id]
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": f"WebSocket error: {e}"
        })

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Serve CSS and JS files directly from root path for compatibility
@app.get("/style.css")
async def get_style():
    with open("static/style.css", "r") as f:
        return Response(content=f.read(), media_type="text/css")

@app.get("/main.js")
async def get_script():
    with open("static/main.js", "r") as f:
        return Response(content=f.read(), media_type="application/javascript")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="localhost")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--backend-host', default="localhost")
    parser.add_argument('--backend-port', default="8000")
    args = parser.parse_args()
    
    # Update chat app with backend connection details
    chat_app.host = args.backend_host
    chat_app.port = args.backend_port
    
    uvicorn.run(app, host=args.host, port=args.port)