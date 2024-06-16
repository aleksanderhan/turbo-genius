from llms import load_huggingface_pipeline, load_with_llama_cpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

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
parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument('--port', action='store', default=8000)
parser.add_argument('--image_generation', action='store_true', default=False)
parser.add_argument('--image_model', action='store', default="sd-community/sdxl-flash")
parser.add_argument('--image_cpu_offload', action='store_true', default=False)
parser.add_argument('--llm_cpu_offload', action='store_true', default=False)
args = parser.parse_args()

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_gguf="./models/Mistral-7B-Instruct-v0.3.Q6_K.gguf"
model_hf="mistralai/Mistral-7B-Instruct-v0.3"


llm = load_with_llama_cpp(model_gguf, callback_manager)
#llm = load_huggingface_pipeline(model_hf, callback_manager)

llm_chain = prompt | llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
for chunk in llm_chain.stream({"question": question}):
    print(chunk, end="", flush=True)



'''

class Agent:

    def __init__(self, llm):
        self.model = llm

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

    def make_title(session: Session):
        messages = session.get_messages()[-2:]
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

    
'''