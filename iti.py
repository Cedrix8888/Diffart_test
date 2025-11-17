from PIL import Image
import torch
import numpy as np
import safetensors.torch as sf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from backend.algorithms.Trans.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from backend.algorithms.Trans.vae import TransparentVAEDecoder, TransparentVAEEncoder
from utils.model import download_model

# Load models
# RealVisXL_V4.0 is a specific version of SDXL
# We use float16 and fp16 for more compatibility and less memory usage
device = "cuda"
sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.float16, variant="fp16")
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

positive_prompt = input("Please enter positive prompt.")
default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'

# Download Model
path_ld_diffusers_sdxl_attn = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors',
    local_path='./models/ld_diffusers_sdxl_attn.safetensors'
)

path_ld_diffusers_sdxl_vae_transparent_encoder = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
    local_path='./models/ld_diffusers_sdxl_vae_transparent_encoder.safetensors'
)

path_ld_diffusers_sdxl_vae_transparent_decoder = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
    local_path='./models/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'
)

# SDP(Scaled Dot-Product Attention)
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Merge weights to fine-tune the original model
sd_offset = sf.load_file("./models/ld_diffusers_sdxl_attn.safetensors")
sd_origin = unet.state_dict()
sd_merged = {
    k: sd_origin[k] + sd_offset[k] if k in sd_offset else sd_origin[k]
    for k in sd_origin.keys()
}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged

# Use the specific VAE
transparent_encoder = TransparentVAEEncoder("./models/ld_diffusers_sdxl_vae_transparent_encoder.safetensors")
transparent_decoder = TransparentVAEDecoder("./models/ld_diffusers_sdxl_vae_transparent_decoder.safetensors")

# Pipelines
pipeline = KDiffusionStableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

with torch.inference_mode():
    guidance_scale = 7.0
    rng = torch.Generator(device=device).manual_seed(12345)
    text_encoder.to(device)
    text_encoder_2.to(device)
    unet.to(device)
    vae.to(device)
    transparent_decoder.to(device)
    transparent_encoder.to(device)
    
    positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(positive_prompt)
    negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(default_negative)

    initial_latent = [np.array(Image.open('./imgs/inputs/causal_cut.png'))]
    initial_latent = transparent_encoder(vae, initial_latent) * 0.18215
    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)
    latents=pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=25,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=guidance_scale,
    ).images


    latents = latents.to(dtype=vae.dtype, device=vae.device) / 0.18215
    result_list = transparent_decoder(vae, latents)

    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f'./imgs/outputs/i2i_{i}_transparent.png', format='PNG')
