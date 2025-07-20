from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler, AutoencoderTiny
from diffusers.utils import make_image_grid

from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output
from typing import List, Callable, Optional, Union

# Load the pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "sd-community/sdxl-flash",
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True
)
pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)

pipeline.enable_vae_tiling()
pipeline.enable_vae_slicing()
pipeline.enable_sequential_cpu_offload()

pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
pipeline.set_progress_bar_config(disable=True)

prompt = "Casino on the beach."
num_inference_steps = 11
guidance_scale = 7.0
target_height = 1024
target_width = 1024

# Initialize the plot outside the callback
fig, ax = plt.subplots(figsize=(target_width/100, target_height/100), dpi=100)
image_plot = None

def visualize_callback(i: int, t: int, latents: torch.Tensor) -> None:
    global image_plot
    image = pipeline.vae.decode(latents / pipeline.vae.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)

    ax.set_title(f"Step {i+1}/{num_inference_steps}")
    
    clear_output(wait=True)
    if image_plot is None:
        image_plot = ax.imshow(image)
        ax.axis('off')
        plt.show(block=False)
    else:
        image_plot.set_data(image)
        fig.canvas.draw()
        fig.canvas.flush_events()

# Run the pipeline with the callback
image = pipeline(
    prompt=prompt,
    height=target_height,
    width=target_width,
    num_inference_steps=num_inference_steps,
    callback=visualize_callback,
    callback_steps=1,
    guidance_scale=guidance_scale,
).images[0]


clear_output(wait=True)
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show(block=True)

image.save(f"sd_{prompt}.png")