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
            text = await self.queue.get()
            if text is None:
                break
            yield text

    def put(self, tokens):
        print(tokens)
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        asyncio.create_task(self.queue.put(' '.join(text)))
    
    async def finish(self):
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
    inputs = tokenizer(prompt, return_tensors="pt")
    task = asyncio.create_task(model.generate(**inputs, streamer=streamer, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9))
    async for text in streamer.stream():
        yield text
    await task

@app.get("/stream")
async def stream(prompt: str = Query(...)):
    return StreamingResponse(generate_response(prompt), media_type="text/plain")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 