import os
import random
import torch
import safetensors.torch as sf
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from vae import TransparentVAEDecoder, TransparentVAEEncoder
# Ensure the utils module contains the download_model function or replace it with the correct module
from utils import download_model # type: ignore

def gen_trans(width: int = 1024,
              height: int = 1024,
              prompt_pos: str = "Create a vivid and detailed scene of a serene beach scene, featuring a young girl with a bright smile, a sunset, and a group of friends. Include elements like waves, sand, the sky, and the sound of the ocean. Make sure to capture the beauty and tranquility of the setting.",
              prompt_neg: str = "face asymmetry, eyes asymmetry, deformed eyes, open mouth"
              ):
    
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("Width and height must be multiples of 8.")
    
    # Load models
    # RealVisXL_V4.0 is a specific version of SDXL
    # We use float16 and fp16 for more compatibility and less memory usage
    device = torch.device("cuda")
    sdxl_name = 'SG161222/RealVisXL_V4.0'
    tokenizer = CLIPTokenizer.from_pretrained(
        sdxl_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        sdxl_name, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder", dtype=torch.float16, variant="fp16")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder_2", dtype=torch.float16, variant="fp16")
    vae = AutoencoderKL.from_pretrained(
        sdxl_name, subfolder="vae", dtype=torch.float16, variant="fp16")
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_name, subfolder="unet", dtype=torch.float16, variant="fp16")

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
    sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
    sd_origin = unet.state_dict()
    sd_merged = {
        k: sd_origin[k] + sd_offset[k] if k in sd_offset else sd_origin[k]
        for k in sd_origin.keys()
    }
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    # Use the specific VAE
    transparent_encoder = TransparentVAEEncoder(path_ld_diffusers_sdxl_vae_transparent_encoder)
    transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder)

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
        seed=random.randint(0, 1000000)
        rng = torch.Generator(device=device).manual_seed(seed)
        text_encoder.to(device) # type: ignore
        text_encoder_2.to(device) # type: ignore
        unet.to(device) # type: ignore
        vae.to(device) # type: ignore
        transparent_decoder.to(device)
        transparent_encoder.to(device)
        
        positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(prompt_pos)
        negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(prompt_neg)

        # (BCHW) -> (BCHW/8)
        initial_latent = torch.zeros(size=(1, 4, height//8, width//8), dtype=unet.dtype, device=unet.device)
        latents_out = pipeline(
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
        )

        result_list = transparent_decoder(vae, latents_out.to(dtype=vae.dtype, device=vae.device)/0.18215)
        image_list = [Image.fromarray(result) for result in result_list]
        return image_list

if __name__ == "__main__":
    # 确保 static 目录存在
    os.makedirs("./static", exist_ok=True)
    # 保存文件
    gen_trans()[0].save("./static/test.png")
