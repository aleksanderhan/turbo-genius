import torch
import flash_attn
import uvicorn
import gc
import asyncio
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model


#model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

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

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


async def generate_token_by_token(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    past_key_values = None

    while True:
        outputs = model(**inputs, past_key_values=past_key_values, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        # Sample or select the next token here
        next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
        
        if next_token in terminators:
            break

        yield next_token.item()

        # Update the input for the next iteration
        inputs = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)
        past_key_values = outputs.past_key_values

async def generate_response(prompt: str):
    async for token in generate_token_by_token(prompt):
        yield tokenizer.decode([token])

@app.get("/stream")
async def stream(prompt: str = Query(...)):
    return StreamingResponse(generate_response(prompt), media_type="text/event-stream")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 