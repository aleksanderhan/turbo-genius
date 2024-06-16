from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, pipeline
import torch


def load_with_llama_cpp(gguf_path, callback_manager):
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=gguf_path,
        max_tokens =1000,
        n_threads = 6,
        temperature= 0.8,
        f16_kv=True,
        n_ctx=28000, 
        n_gpu_layers=24,
        callback_manager=callback_manager, 
        verbose=True,
        top_p=0.75,
        top_k=40,
        repeat_penalty = 1.1,
        streaming=True,
        model_kwargs={
                'mirostat': 2,
        },
    )
    return llm

def load_huggingface_pipeline(model_path, callback_manager):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=False
    )
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        config=config,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe, callback_manager=callback_manager)
    return llm


