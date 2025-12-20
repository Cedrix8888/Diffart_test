from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch

# Load with optimizations
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
lora_path = "lora/aesthetic_anime_v1s.safetensors"
pipe.load_lora_weights(lora_path, adapter_name="aesthetic_anime", prefix=None)
pipe.set_adapters(adapter_names=["aesthetic_anime"], adapter_weights=[0.8])
pipe.fuse_lora()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# Call the pipeline with optimized settings
result = pipe(
    prompt,
    num_inference_steps=30,  # Reduce steps for faster generation (default is 50)
    guidance_scale=7.5
)

image = result.images[0] # type: ignore
image.save("astronaut_jungle.png")
