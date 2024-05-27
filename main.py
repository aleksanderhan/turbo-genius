import torch
import asyncio
import flash_attn
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextStreamer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model


model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
#model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

class AsyncTextStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.queue = asyncio.Queue()

    async def stream(self):
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token

    async def put(self, token_id):
        print("put", token_id)
        token = self.tokenizer.batch_decode(token_id, skip_special_tokens=True)
        asyncio.create_task(self.queue.put(token))
    
    async def end(self):
        await self.queue.put(None)


app = FastAPI()

config = AutoConfig.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    config=config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
streamer = AsyncTextStreamer(tokenizer)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

async def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    task = asyncio.create_task(model.generate(**inputs, streamer=streamer, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9))
    async for token in streamer.stream():
        yield token
    await task

@app.get("/stream")
async def stream(prompt: str = Query(...)):
    return StreamingResponse(generate_response(prompt), media_type="text/plain")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 