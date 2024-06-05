import torch
import flash_attn
import uvicorn
import gc
import asyncio
import argparse
from fastapi import FastAPI, WebSocket, Depends
from threading import Thread
from sqlalchemy.orm import Session as DBSession

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model

from session import Session, SessionManager, get_db, SessionDB

app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = AutoConfig.from_pretrained(args.model)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map='auto',
    config=config,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(args.model)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids(""),
]

summarizer = pipeline(task="summarization", model="facebook/bart-large", min_length=2, max_length=10)


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

    # Run the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Start streaming tokens
    async for token in stream_tokens(streamer):
        yield token

    thread.join()

async def make_title(session: Session):
    messages = session.get_messages()[1:3]
    prompt = "\n".join([message["content"] for message in messages])
    return summarizer(prompt)

def make_prompt(session: Session):
    inputs = tokenizer.apply_chat_template(
        session.get_messages(),
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True
    )
    num_tokens = inputs.shape[-1]
    print(f"Number of tokens (session {session.id}): ", num_tokens)
    if num_tokens > int(config.max_position_embeddings * 0.9):
        session.truncate_messages()
        return make_prompt(session)
    else:
        return tokenizer.apply_chat_template(
            session.get_messages(),
            add_generation_prompt=True,
            tokenize=False
        )

@app.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: int, db: DBSession = Depends(get_db)):
    await websocket.accept()
    message = await websocket.receive_text()
    session = session_manager.get_session(session_id, db)
    session.add_user_message(message)
    db_session = db.query(SessionDB).filter(SessionDB.id == session.id).first()
    db_session.messages = str(session.messages)
    db.add(db_session)
    db.commit()
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
        db_session.messages = str(session.messages)
        db.add(db_session)
        db.commit()
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
    summary_response = await make_title(session)
    session.title = summary_response[0]["summary_text"]
    db_session = db.query(SessionDB).filter(SessionDB.id == session.id).first()
    db_session.title = session.title
    db.add(db_session)
    db.commit()
    return session.title

if __name__ == "__main__":
    session_manager = SessionManager()
    uvicorn.run(app, host="0.0.0.0", port=8000)
