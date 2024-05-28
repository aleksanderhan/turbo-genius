import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(args.model)

dummy_input = tokenizer("This is a test input.", return_tensors="pt").input_ids.long()
torch.onnx.export(
    model,
    dummy_input,
    "model/model.onnx",
    verbose=True,
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
    opset_version=14
)
