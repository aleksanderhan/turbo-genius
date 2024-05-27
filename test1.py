from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()\

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    try:
        await websocket.accept()
        print("WebSocket connection accepted")
        # Proceed with your intended operations
    except Exception as e:
        print(f"WebSocket connection failed: {e}")
        raise e
    finally:
        await websocket.close()
        print("WebSocket connection closed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 