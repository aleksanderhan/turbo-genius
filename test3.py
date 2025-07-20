import torch
from torch.amp import autocast
from diffusers import PixArtSigmaPipeline, DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
import inspect
from matplotlib.widgets import RectangleSelector
import numpy as np


# ── CONFIG ─────────────────────────────────────────────────────────────────────
model_id            = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
height, width       = 1024, 1024
num_inference_steps = 11    # steps per iteration
guidance_scale      = 7.0
callback_steps      = 1
device              = torch.device("cuda")
generator           = torch.Generator(device=device).manual_seed(42)
do_cfg              = guidance_scale > 1.0

# ── LOAD PIPELINE ──────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
pipe = PixArtSigmaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="balanced",
    offload_folder="offload",
    offload_state_dict=True,
    use_safetensors=True,
)
pipe.set_progress_bar_config(disable=True)

# ── CALLBACK & PLOT SETUP ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
image_plot = None
iteration = 0  # ensure defined early


def visualize_callback(i: int, t: int, latents: torch.Tensor) -> None:
    """Display the current state of the image generation process."""
    global image_plot
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    clear_output(wait=True)
    
    current_step = i + 1  # 1-based index for display
    ax.set_title(f"Iteration {iteration}: Step {current_step}/{num_inference_steps}")
    
    if image_plot is None:
        image_plot = ax.imshow(image)
        ax.axis('off')
        plt.show(block=False)
    else:
        image_plot.set_data(image)
        fig.canvas.draw()
        fig.canvas.flush_events()


# ── PREPARE INITIAL LATENTS ────────────────────────────────────────────────────
def initialize_latents():
    """Initialize latents with the proper noise scale."""
    latent_channels = pipe.transformer.config.in_channels
    # Create properly scaled initial noise
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    return latents


# ── DENOISING PHASE FUNCTION ───────────────────────────────────────────────────
def denoise_phase(prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, latents, strength=1.0):
    """Run a single denoising phase with proper handling of scheduler and noise levels."""
    # Set timesteps for this phase
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # If we're not starting from scratch (strength < 1.0), adjust starting timestep
    if strength < 1.0:
        init_timestep = int(num_inference_steps * (1.0 - strength))
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = timesteps[init_timestep:]
    
    # Combine prompt embeddings and attention masks for classifier-free guidance
    if do_cfg:
        combined_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        combined_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
    else:
        combined_prompt_embeds = prompt_embeds
        combined_attention_mask = prompt_attention_mask
    
    # Main denoising loop
    for i, t in enumerate(timesteps):
        # Prepare the model input (double for classifier-free guidance)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Prepare timestep tensor
        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.int64, device=latent_model_input.device)
        
        current_timestep = current_timestep.expand(latent_model_input.shape[0])

        # Run the transformer to predict noise
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        noise_pred = pipe.transformer(
            latent_model_input,
            encoder_hidden_states=combined_prompt_embeds,
            encoder_attention_mask=combined_attention_mask,
            timestep=current_timestep,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # Apply classifier-free guidance
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Handle learned sigma if present
        if pipe.transformer.config.out_channels // 2 == pipe.transformer.config.in_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]

        # Compute the previous image in the diffusion process
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # Update visualization
        if (i + 1) % callback_steps == 0:
            visualize_callback(i, t, latents)
    
    return latents


# ── INTERACTIVE REFINEMENT LOOP ────────────────────────────────────────────────
prompt = ""
latents = initialize_latents()  # Start with proper initial noise
cumulative_strength = 1.0  # Start with full denoising for first iteration
combined_prompt = ""

while True:
    new = input("Add to prompt (blank to finish): ")
    if not new.strip():
        break

    # Update prompt and track iteration
    combined_prompt += " " + new.strip() if combined_prompt else new.strip()
    prompt = combined_prompt  # Use the full accumulated prompt each time
    negative_prompt = ""
    iteration += 1
    
    print(f"Current prompt: \"{prompt}\"")

    # For subsequent iterations, use lower strength to preserve more of the existing image
    if iteration > 1:
        cumulative_strength = 0.65  # Fixed mid-level strength for refinements
    
    # Encode the current prompt
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            device="cpu",  # Encode on CPU first
            clean_caption=True,
            max_sequence_length=384,
        )

    # Move tensors to device
    prompt_embeds = prompt_embeds.to(device, non_blocking=True)
    prompt_attention_mask = prompt_attention_mask.to(device, non_blocking=True)
    negative_prompt_embeds = negative_prompt_embeds.to(device, non_blocking=True)
    negative_prompt_attention_mask = negative_prompt_attention_mask.to(device, non_blocking=True)

    # Run the denoising phase with the current strength setting
    with torch.no_grad():
        with autocast("cuda", dtype=torch.float16):
            # Apply true continuity with Karras scheduler
            latents = denoise_phase(
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
                latents,
                strength=cumulative_strength
            )

            # Save the current latents to maintain continuity
            torch.cuda.empty_cache()


# ── FINAL DECODE ───────────────────────────────────────────────────────────────
with torch.no_grad():
    with autocast("cuda", dtype=torch.float16):
        final_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        final_image = (final_image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]

out = Image.fromarray((final_image * 255).round().astype("uint8"))
safe_filename = prompt.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')[:100]
out.save(f"{safe_filename}.png")
print(f"Saved final image as: {safe_filename}.png")
