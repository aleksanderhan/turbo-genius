from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import ktransformers
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeForCausalLM
from ktransformers.models.configuration_qwen3_moe import Qwen3MoeConfig

# Load model and tokenizer (replace with Qwen2/Qwen3 model ID)
model_id = "Qwen/Qwen1.5-0.5B"  # or e.g., "Qwen/Qwen2-1.8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)


# Patch the model with ktransformers
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, "qwen3moe_optimize.yaml", model_id, config)




# Generate text
prompt = "Explain quantum entanglement in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
