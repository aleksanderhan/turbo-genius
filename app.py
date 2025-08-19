import torch
import flash_attn
import uvicorn
import gc
import asyncio
import argparse
import io
import json
import re
import sys
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from threading import Thread
from sqlalchemy.orm import Session as DBSession
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer, pipeline
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler, AutoencoderTiny
from typing import Dict
from huggingface_hub import hf_hub_download


from session import Session, SessionManager, SessionDB, SessionImageDB, get_db

app = FastAPI()
session_manager = SessionManager()

# Store active WebSocket connections for the web interface
web_connections: Dict[str, WebSocket] = {}

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_repo', action='store', default="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF")
parser.add_argument('--model_filename', action='store', default="Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")
parser.add_argument('--tokenizer_repo', action='store', default="Qwen/Qwen3-30B-A3B-Instruct-2507")
parser.add_argument('--port', action='store', default=8000)
parser.add_argument('--image_generation', action='store_true', default=False)
parser.add_argument('--image_model', action='store', default="sd-community/sdxl-flash")
parser.add_argument('--image_cpu_offload', action='store_true', default=False)
args = parser.parse_args()


base_path = "./models/"

try:
    local_path = hf_hub_download(
        repo_id=args.model_repo,
        filename=args.model_filename,
        local_dir=base_path,  # Optional: Specify a local directory to save the file
        local_dir_use_symlinks=False,  # Recommended for avoiding potential issues
        force_download=False,
    )
    print(f"GGUF file downloaded to: {local_path}")
except Exception as e:
    print(f"Error downloading file: {e}")
    sys.exit(0)


from llama_cpp import Llama

config = AutoConfig.from_pretrained(args.tokenizer_repo)

model = Llama(
    model_path=base_path + args.model_filename,    
    n_ctx=16384,
    n_threads=os.cpu_count(),
    n_batch=1,
    n_gpu_layers=15,
    verbose=True,
)



tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_repo)
terminators = [
    tokenizer.eos_token_id,
    #tokenizer.convert_tokens_to_ids(""),
]


summarizer = pipeline(
    task="summarization", 
    model="facebook/bart-large-cnn", 
    min_length=2, 
    max_length=10,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    device="cpu"
)

if args.image_generation:
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(args.image_model, torch_dtype=torch.float16)
    sdxl_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(sdxl_pipe.scheduler.config, timestep_spacing="trailing")
    sdxl_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
    sdxl_pipe.enable_vae_tiling()
    sdxl_pipe.enable_vae_slicing()
    if args.image_cpu_offload:
        sdxl_pipe.enable_sequential_cpu_offload()

# AI generation functions (from your original server)
async def stream_tokens(streamer: TextIteratorStreamer):
    for token in streamer:
        yield token
    yield None

async def generate_response(prompt: str):
    torch.cuda.empty_cache()
    gc.collect()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) 
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_length": config.max_position_embeddings,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async for token in stream_tokens(streamer):
        yield token

    thread.join()

def make_title(session: Session):
    messages = session.get_messages()[-2:]
    prompt = "\n".join([message["content"] for message in messages])
    return summarizer(prompt)

def make_prompt(session: Session):
    inputs = tokenizer.apply_chat_template(
        session.get_messages(),
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        enable_thinking=False
    )
    num_tokens = inputs.shape[-1]
    if num_tokens > int(config.max_position_embeddings * 0.9):
        session.truncate_messages()
        return make_prompt(session)
    else:
        return tokenizer.apply_chat_template(
            session.get_messages(),
            add_generation_prompt=True,
            tokenize=False
        )

def generate_image(session_id: int, prompt: str, db: DBSession):
    torch.cuda.empty_cache()
    gc.collect()
    image = sdxl_pipe(prompt, num_inference_steps=6, guidance_scale=3).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    image_db = SessionImageDB(session_id=session_id, image=img_byte_arr)
    db.add(image_db)
    db.commit()
    db.refresh(image_db)

    image_url = f'<img class="scaled" src="http://localhost:{args.port}/image/{image_db.id}" alt="{prompt}" />'    
    return image_url

# Original WebSocket endpoint for desktop client compatibility
@app.websocket("/stream/{session_id}")
async def stream_original(websocket: WebSocket, session_id: int, db: DBSession = Depends(get_db)):
    await websocket.accept()
    message = await websocket.receive_text()
    session = session_manager.get_session(session_id, db)
    
    if message.startswith("image:"):
        prompt = message[len("image:"):].strip()
        session.add_user_message(message)
        image_tag = generate_image(session_id, prompt, db)
        await websocket.send_text(image_tag)
        await asyncio.sleep(0.01)
        session.add_assistant_message(image_tag)
        session_manager.save_session(session, db)                    
    else:
        session.add_user_message(message)
        session_manager.save_session(session, db)            
        prompt = make_prompt(session)
        completion = ""
        try:
            async for token in generate_response(prompt):
                if token is None:
                    break
                completion += token
                await websocket.send_text(token)
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            session.add_assistant_message(completion)
            session_manager.save_session(session, db)            
            await websocket.close()

# New WebSocket endpoint for web client
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: DBSession = Depends(get_db)):
    await websocket.accept()
    connection_id = id(websocket)
    web_connections[connection_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "send_message":
                session_id = data.get("session_id")
                message = data.get("message")
                
                # Create new session if needed
                if not session_id:
                    session = session_manager.get_new_session(db)
                    session_id = session.id
                    await websocket.send_json({
                        "type": "session_created",
                        "session_id": session_id
                    })
                
                if session_id and message:
                    session = session_manager.get_session(session_id, db)
                    
                    # Handle image generation
                    if message.startswith("image:"):
                        prompt = message[len("image:"):].strip()
                        session.add_user_message(message)
                        image_tag = generate_image(session_id, prompt, db)
                        await websocket.send_json({
                            "type": "message",
                            "role": "assistant",
                            "content": image_tag
                        })
                        session.add_assistant_message(image_tag)
                        session_manager.save_session(session, db)
                    else:
                        # Handle text generation
                        session.add_user_message(message)
                        session_manager.save_session(session, db)            
                        prompt = make_prompt(session)
                        completion = ""
                        
                        try:
                            async for token in generate_response(prompt):
                                if token is None:
                                    break
                                completion += token
                                await websocket.send_json({
                                    "type": "message",
                                    "role": "assistant",
                                    "content": token
                                })
                        except Exception as e:
                            print(f"Error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Generation error: {e}"
                            })
                        finally:
                            session.add_assistant_message(completion)
                            session_manager.save_session(session, db)
                    
            elif action == "load_session":
                session_id = data.get("session_id")
                session = session_manager.get_session(session_id, db)
                
                # Process image tags for web client
                session_data = session.to_dict()
                for message in session_data["messages"]:
                    if message["role"] in ["user", "assistant"]:
                        img_tag_pattern = r'<img\b[^>]*>'
                        if re.search(img_tag_pattern, message["content"]):
                            message["content"] = message["content"].replace(
                                "localhost", "localhost"
                            ).replace(
                                str(args.port), str(args.port)
                            )
                
                await websocket.send_json({
                    "type": "session_loaded",
                    "data": session_data
                })
                    
            elif action == "delete_session":
                session_id = data.get("session_id")
                session_manager.remove_session(session_id, db)
                db.commit()
                await websocket.send_json({
                    "type": "session_deleted",
                    "success": True,
                    "session_id": session_id
                })
                
            elif action == "get_sessions":
                sessions = session_manager.get_session_list(db)
                await websocket.send_json({
                    "type": "sessions_list",
                    "sessions": sessions
                })
                
            elif action == "generate_title":
                session_id = data.get("session_id")
                session = session_manager.get_session(session_id, db)
                summary_response = make_title(session)
                title = summary_response[0]["summary_text"]
                
                # Update session title
                session.title = title
                db_session = db.query(SessionDB).filter(SessionDB.id == session.id).first()
                db_session.title = title
                db.add(db_session)
                db.commit()
                
                await websocket.send_json({
                    "type": "session_title_updated",
                    "session_id": session_id,
                    "title": title
                })
                
    except WebSocketDisconnect:
        if connection_id in web_connections:
            del web_connections[connection_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"WebSocket error: {e}"
        })

# REST API endpoints (original functionality)
@app.get("/session")
async def get_session(db: DBSession = Depends(get_db)):
    session = session_manager.get_new_session(db)
    return session.id

@app.get("/session/{session_id}")
async def get_session_data(session_id: int, db: DBSession = Depends(get_db)):
    session = session_manager.get_session(session_id, db)
    return session

@app.get("/session-list")
async def get_session_list(db: DBSession = Depends(get_db)):
    sessions = session_manager.get_session_list(db)
    return sessions

@app.delete("/session/{session_id}")
async def delete_session(session_id: int, db: DBSession = Depends(get_db)):
    session_manager.remove_session(session_id, db)
    db.commit()
    return

@app.get("/session/{session_id}/title")
async def get_session_title(session_id: int, db: DBSession = Depends(get_db)):
    session = session_manager.get_session(session_id, db)
    summary_response = make_title(session)
    session.title = summary_response[0]["summary_text"]
    db_session = db.query(SessionDB).filter(SessionDB.id == session.id).first()
    db_session.title = session.title
    db.add(db_session)
    db.commit()
    return session.title

@app.get("/image/{image_id}")
async def get_image(image_id: int, db: DBSession = Depends(get_db)):
    image_db = db.query(SessionImageDB).filter(SessionImageDB.id == image_id).first()    
    img_byte_arr = io.BytesIO(image_db.image)
    return Response(img_byte_arr.getvalue(), media_type="image/png")

# Web interface endpoints
@app.get("/")
async def get_index():
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1><p>Please create static/index.html</p>")

@app.get("/style.css")
async def get_style():
    try:
        with open("static/style.css", "r") as f:
            return Response(content=f.read(), media_type="text/css")
    except FileNotFoundError:
        return Response(content="/* CSS file not found */", media_type="text/css")

@app.get("/main.js")
async def get_script():
    try:
        with open("static/main.js", "r") as f:
            return Response(content=f.read(), media_type="application/javascript")
    except FileNotFoundError:
        return Response(content="// JS file not found", media_type="application/javascript")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":        
    print(f"Starting unified AI chat server on port {args.port}")
    print(f"Web interface: http://localhost:{args.port}")
    print(f"Model: {args.model_repo}")
    if args.image_generation:
        print(f"Image generation enabled with model: {args.image_model}")
    uvicorn.run(app, host="0.0.0.0", port=int(args.port))