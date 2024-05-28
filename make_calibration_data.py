import numpy as np
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', action='store', default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

# Example sentences
sentences = [
    "Tell me about the connection between the halting problem and the spectral gap problem.",
    "Tell me about the limits of computatability.",
    "Does TensorRT require representative data for calibration?",
    "What is the difference between Flash Attention and Flash Attention 2?",
    "Write me a python script to do supervised fine-tuning of a large language model!",
]

# Tokenize sentences and save as .npy files
for i, sentence in enumerate(sentences):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy()
    np.save(f"calibration_data_{i}.npy", input_ids)
