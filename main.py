import torch
import flash_attn
import uvicorn
import gc
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TextIteratorStreamer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model


#model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify the correct origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print(f"Initial input_ids shape: {inputs['input_ids'].shape}")  # Debugging statement for initial shape

    past_key_values = None

    while True:
        torch.cuda.empty_cache()
        gc.collect()
        outputs = model(**inputs, past_key_values=past_key_values, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        # Sample the next token from the logits
        next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)

        # Check for termination conditions
        if next_token.item() in terminators:
            break

        # Correctly reshape next_token for concatenation
        next_token = next_token.view(1, 1)  # Changing from [1] to [1, 1]

        # Concatenate the new token to the existing sequence
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)

        past_key_values = outputs.past_key_values

        # Decode the token to string and yield
        decoded_token = tokenizer.decode(next_token.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(decoded_token)
        yield decoded_token


async def generate_response(prompt: str):
    try:
        async for token in generate_token_by_token(prompt):
            yield token
    except asyncio.CancelledError:
        pass  # Handle client disconnect gracefully


@app.websocket("/stream")
async def stream(websocket: WebSocket, prompt: str):
    await websocket.accept()  # Accept the WebSocket connection
    try:
        async for token in generate_token_by_token(prompt):
            await websocket.send_text(token)  # Send each token as soon as it's generated
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close() 



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 