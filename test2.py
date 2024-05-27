@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Test message")
    await websocket.close()
