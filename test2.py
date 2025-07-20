import torch
from diffusers import PixArtSigmaPipeline, DPMSolverSinglestepScheduler, LCMScheduler, PixArtSigmaPAGPipeline
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ── CONFIG ─────────────────────────────────────────────────────────────────────
model_id            = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
prompt              = "Casino on the beach."
height, width       = 1024, 1024
num_inference_steps = 20
guidance_scale      = 7.0
callback_steps      = 1

# ── LOAD PIPELINE WITH OFFLOADING ──────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
pipeline = PixArtSigmaPAGPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="balanced",            # auto-place text_encoder on CPU, transformer+VAE on GPU
    offload_folder="offload",         # where to spill intermediate weights
    offload_state_dict=True,          # keep only active modules in memory
    use_safetensors=True,
)
pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing="trailing"
)
#pipeline.enable_xformers_memory_efficient_attention()
pipeline.set_progress_bar_config(disable=True)


# ── SETUP PLOT ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
image_plot = None

def visualize_callback(step, timestep, latents):
    global image_plot
    # decode & normalize
    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)

    ax.set_title(f"Step {step+1}/{num_inference_steps}")

    clear_output(wait=True)
    if image_plot is None:
        image_plot = ax.imshow(image)
        ax.axis('off')
        plt.show(block=False)
    else:
        image_plot.set_data(image)
        fig.canvas.draw()
        fig.canvas.flush_events()

# ── GENERATE WITH CALLBACK ────────────────────────────────────────────────────
image = pipeline(
    prompt=prompt,
    height=height, width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    callback=visualize_callback,
    callback_steps=callback_steps,
).images[0]

# final display
clear_output(wait=True)
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show()

image.save(f"pa_{prompt}.png")