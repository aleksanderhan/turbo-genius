import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(args.model)

dummy_input = tokenizer("This is a test input.", return_tensors="pt").input_ids
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
    opset_version=14
)
