from llms import load_huggingface_pipeline, load_with_llama_cpp, load_with_vllm
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

import torch
import flash_attn
import uvicorn
import gc
import asyncio
import argparse
import io
from fastapi import FastAPI, WebSocket, Depends
from fastapi.responses import Response
from threading import Thread
from sqlalchemy.orm import Session as DBSession
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer, pipeline
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler, AutoencoderTiny

from session import Session, SessionManager, SessionDB, SessionImageDB, get_db

app = FastAPI()
session_manager = SessionManager()

parser = argparse.ArgumentParser()
parser.add_argument('--repo', action='store', default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument('--file', action='store', default=None)
parser.add_argument('--awq', action='store_true', default=False)
parser.add_argument('--port', action='store', default=8000)
parser.add_argument('--image_generation', action='store_true', default=False)
parser.add_argument('--image_model', action='store', default="sd-community/sdxl-flash")
parser.add_argument('--image_cpu_offload', action='store_true', default=False)
parser.add_argument('--llm_cpu_offload', action='store_true', default=False)
args = parser.parse_args()



if args.file is not None and args.file.endswith(".gguf"):
    llm = load_with_llama_cpp(args.repo, args.file)
elif args.awq:
    llm = load_with_vllm(args.repo)
else:
    llm = load_huggingface_pipeline(args.repo)


question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

summarizer = pipeline(
    task="summarization", 
    model="facebook/bart-large-cnn", 
    min_length=2, 
    max_length=10,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

if args.image_generation:
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(args.image_model, torch_dtype=torch.float16)
    sdxl_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(sdxl_pipe.scheduler.config, timestep_spacing="trailing")
    sdxl_pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
    sdxl_pipe.enable_vae_tiling()
    sdxl_pipe.enable_vae_slicing()
    if args.image_cpu_offload:
        sdxl_pipe.enable_sequential_cpu_offload()



async def generate_response(message: str, session: Session):
    template = """You have access to tools that you can use to comply with the user requests.
    
### Conversation history so far:
{history}
    
### User request:
{request}

### Assistant response:
"""
    
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm
    async for chunk in llm_chain.astream({"history": make_history(session), "request": message}):
        yield chunk

def make_history(session: Session):
    messages = session.get_messages()
    return "\n".join([message['role'] + "\n" + message["content"] + "\n\n" for message in messages])

def make_title(session: Session):
    messages = session.get_messages()[-2:]
    prompt = "\n".join([message["content"] for message in messages])
    return summarizer(prompt)

def generate_image(session_id: int, prompt: str, db: DBSession):
    torch.cuda.empty_cache()
    gc.collect()
    image = sdxl_pipe(prompt, num_inference_steps=6, guidance_scale=3).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Save the image to the database
    image_db = SessionImageDB(session_id=session_id, image=img_byte_arr)
    db.add(image_db)
    db.commit()
    db.refresh(image_db)

    # Create an image URL
    image_url = f'<img class="scaled" src="http://<host>:<port>/image/{image_db.id}" alt="{prompt}" />'    
    return image_url

@app.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: int, db: DBSession = Depends(get_db)):
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
        completion = ""
        try:
            async for token in generate_response(message, session):
                if token is None:
                    break
                completion += token
                await websocket.send_text(token)
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            session.add_user_message(message)
            session.add_assistant_message(completion)
            session_manager.save_session(session, db)            
            await websocket.close()


@app.get("/session")
async def get_session(db: DBSession = Depends(get_db)):
    session = session_manager.get_new_session(db)
    return session.id

@app.get("/session/{session_id}")
async def get_session(session_id: int, db: DBSession = Depends(get_db)):
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

if __name__ == "__main__":        
    uvicorn.run(app, host="0.0.0.0", port=args.port)

