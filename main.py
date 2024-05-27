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
    print(f"Initial input_ids shape: {inputs['input_ids'].shape}")  # Debugging statement for initial shape

    past_key_values = None

    while True:
        outputs = model(**inputs, past_key_values=past_key_values, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        print(f"Logits shape: {next_token_logits.shape}")  # Debugging statement for logits shape

        # Sample the next token from the logits
        next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
        print(f"Sampled next_token: {next_token} (shape: {next_token.shape})")  # Debugging sampled token

        # Check for termination conditions
        if next_token.item() in terminators:
            print("Terminator found, stopping generation.")  # Debugging terminator condition
            break

        # Correctly reshape next_token for concatenation
        next_token = next_token.view(1, 1)  # Changing from [1] to [1, 1]
        print(f"Reshaped next_token for concatenation: {next_token} (shape: {next_token.shape})")

        # Concatenate the new token to the existing sequence
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)
        print(f"Updated input_ids shape: {inputs['input_ids'].shape}")  # Debugging updated input_ids shape

        past_key_values = outputs.past_key_values

        # Decode the token to string and yield
        decoded_token = tokenizer.decode(next_token.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"Decoded token: {decoded_token}")  # Debugging decoded token
        yield decoded_token


async def generate_response(prompt: str):
    try:
        async for token in generate_token_by_token(prompt):
            yield token
    except asyncio.CancelledError:
        pass  # Handle client disconnect gracefully

@app.get("/stream")
async def stream(prompt: str = Query(...)):
    return StreamingResponse(generate_response(prompt), media_type="text/event-stream")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 