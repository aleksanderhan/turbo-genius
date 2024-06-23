from langchain_community.llms import LlamaCpp, VLLM
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from transformers import pipeline
import torch
from huggingface_hub import hf_hub_download, snapshot_download

def load_with_llama_cpp(repo, gguf_file):
    hf_hub_download(
        repo_id=repo, 
        filename=gguf_file, 
        resume_download=True,
        local_dir="./models",
    )
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="./models/" + gguf_file,
        max_tokens =1000,
        n_threads = 6,
        temperature= 0.8,
        f16_kv=True,
        n_ctx=28000, 
        n_gpu_layers=-2,
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

def load_huggingface_pipeline(repo):
    snapshot_download(repo, resume_download=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=False
    )
    config = AutoConfig.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        device_map='auto',
        config=config,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(repo)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=8096)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def load_with_vllm(repo):
    snapshot_download(repo, resume_download=True)
    llm = VLLM(
        model=repo,
        trust_remote_code=True,  # mandatory for hf models
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        vllm_kwargs={"quantization": "AWQ", "gpu_memory_utilization": 0.99}
    )
    return llm
