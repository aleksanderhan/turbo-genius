import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler, AutoencoderTiny
import time

# Load model.
pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
#pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_sequential_cpu_offload()



print("sleeping")
time.sleep(10)

# Image generation.
pipe("a happy dog, sunny day, realism", num_inference_steps=7, guidance_scale=3).images[0].save("output.png")
