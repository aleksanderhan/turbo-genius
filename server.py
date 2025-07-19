import torch
#import flash_attn
import uvicorn
import gc
import asyncio
import argparse
import io, sys, os
import traceback
from fastapi import FastAPI, WebSocket, Depends
from fastapi.responses import Response
from threading import Thread
from sqlalchemy.orm import Session as DBSession
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    TextIteratorStreamer, 
)
from huggingface_hub import hf_hub_download
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
from ktransformers.models.configuration_qwen3_moe import Qwen3MoeConfig
from ktransformers.models.custom_cache import StaticCache
from ktransformers.server.utils.create_interface import create_interface, get_interface
from ktransformers.server.config.config import Config

from session import Session, SessionManager, SessionDB, get_db


END_REASONING = "</think>"

app = FastAPI()
session_manager = SessionManager()

parser = argparse.ArgumentParser()
parser.add_argument('--model_repo', action='store', default="unsloth/Qwen3-30B-A3B-GGUF")
parser.add_argument('--model_filename', action='store', default="Qwen3-30B-A3B-Q4_K_M.gguf")
parser.add_argument('--tokenizer_repo', action='store', default="Qwen/Qwen3-30B-A3B")
parser.add_argument('--optimize_config', action='store', default='./qwen3moe_optimize.yaml')
parser.add_argument('--port', action='store', default=8000)
args = parser.parse_args()


try:
    local_path = hf_hub_download(
        repo_id=args.model_repo,
        filename=args.model_filename,
        local_dir="./models",  # Optional: Specify a local directory to save the file
        local_dir_use_symlinks=False,
        force_download=False,
    )
    print(f"GGUF file downloaded to: {local_path}")
except Exception as e:
    print(f"Error downloading file: {e}")
    sys.exit(0)


model_config = Qwen3MoeConfig.from_pretrained(args.model_repo)
print(model_config)

kconfig = Config()
kconfig.load()
print(kconfig)

cache = StaticCache(model_config, max_batch_size=1, max_cache_len=8192, device="cuda")


#with torch.device("meta"):
#    model = Qwen3MoeForCausalLM(model_config, cache)
#optimize_and_load_gguf(model, args.optimize_config, local_path, config)
#print(model)

create_interface(kconfig, kconfig)
interface = get_interface()
print(interface)

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
    min_p=0.05,
)


async def stream_tokens(streamer: TextIteratorStreamer):
    for token in streamer:
        yield token
    yield None

async def generate_response(prompt: str, enable_thinking=False):
    torch.cuda.empty_cache()
    gc.collect()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) 
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "do_sample": True,
        "temperature": 0.6 if enable_thinking else 0.7,
        "top_p": 0.95 if enable_thinking else 0.8,
        "min_p": 0,
        "top_k": 20,
        "max_new_tokens": model_config.max_position_embeddings - len(inputs["input_ids"]),
        "use_cache": True,
    }

    # Run the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Start streaming tokens
    async for token in stream_tokens(streamer):
        yield token

    thread.join()

def make_title(session: Session):
    messages = session.get_messages()[-2:]
    prompt = "\n".join([message["content"] for message in messages])
    return summarizer(prompt)

def make_prompt(session: Session, enable_thinking=False):
    inputs = tokenizer.apply_chat_template(
        session.get_messages(),
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    num_tokens = inputs.shape[-1]
    if num_tokens > int(model_config.max_position_embeddings * 0.8):
        session.truncate_messages()
        return make_prompt(session)
    else:
        return tokenizer.apply_chat_template(
            session.get_messages(),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )


@app.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: int, db: DBSession = Depends(get_db)):
    await websocket.accept()
    enable_thinking = websocket.query_params.get("reason")
    message = await websocket.receive_text()
    session = session_manager.get_session(session_id, db)
    
    session.add_user_message(message)
    session_manager.save_session(session, db)            
    prompt = make_prompt(session, enable_thinking)
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
        traceback.print_exc()
    finally:
        conclusion = completion.split(END_REASONING)[-1]
        session.add_assistant_message(conclusion)
        session_manager.save_session(session, db)            
        await websocket.close()

@app.get("/session")
async def get_session(db: DBSession = Depends(get_db)):
    system_message = "You are a helpful assistant"
    session = session_manager.get_new_session(db, system_message)
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
    #summary_response = #make_title(session)
    session.title = "title" #summary_response[0]["summary_text"]
    db_session = db.query(SessionDB).filter(SessionDB.id == session.id).first()
    db_session.title = session.title
    db.add(db_session)
    db.commit()
    return session.title


if __name__ == "__main__":        
    uvicorn.run(app, host="0.0.0.0", port=args.port)
