import os
import random
import torch
import safetensors.torch as sf
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from vae import TransparentVAEDecoder, TransparentVAEEncoder
from utils import download_model  # 确保utils模块包含download_model函数


class TransparentImageGenerator:
    def __init__(self, sdxl_name: str = "SG161222/RealVisXL_V4.0", device: str = "cuda"):
        """初始化透明图像生成器,加载基础模型和透明VAE组件"""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sdxl_name = sdxl_name
        self._load_base_models()  # 加载SDXL基础模型（文本编码器、VAE、UNet)
        self._load_transparent_vae()  # 加载透明图像专用VAE编码器/解码器
        self._init_pipeline()  # 初始化KDiffusion pipeline
        print(f"Generator initialized on {self.device}")

    def _load_base_models(self):
        """加载SDXL基础模型(文本编码器、Tokenizer、VAE、UNet)"""
        # 1. 加载Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sdxl_name, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.sdxl_name, subfolder="tokenizer_2")

        # 2. 加载文本编码器（float16节省显存)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.sdxl_name, subfolder="text_encoder", dtype=torch.float16, variant="fp16"
        ).to(self.device) # type: ignore
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            self.sdxl_name, subfolder="text_encoder_2", dtype=torch.float16, variant="fp16"
        ).to(self.device) # type: ignore

        # 3. 加载VAE（原始SDXL VAE)
        self.vae = AutoencoderKL.from_pretrained(
            self.sdxl_name, subfolder="vae", torch_dtype=torch.float16, variant="fp16"
        ).to(self.device) # type: ignore
        self.vae.set_attn_processor(AttnProcessor2_0())  # 使用SDP注意力加速

        # 4. 加载UNet并合并透明生成权重
        self.unet = UNet2DConditionModel.from_pretrained(
            self.sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
        ).to(self.device) # type: ignore
        self.unet.set_attn_processor(AttnProcessor2_0())
        # 下载并合并LayerDiffuse UNet偏移权重（论文中潜透明度所需)
        unet_offset_path = download_model(
            url="https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors",
            local_path="./models/ld_diffusers_sdxl_attn.safetensors"
        )
        sd_offset = sf.load_file(unet_offset_path)
        sd_origin = self.unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] if k in sd_offset else sd_origin[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged  # 释放显存

    def _load_transparent_vae(self):
        """加载论文中定义的透明VAE编码器/解码器"""
        # 下载透明VAE权重
        encoder_path = download_model(
            url="https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors",
            local_path="./models/ld_diffusers_sdxl_vae_transparent_encoder.safetensors"
        )
        decoder_path = download_model(
            url="https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors",
            local_path="./models/ld_diffusers_sdxl_vae_transparent_decoder.safetensors"
        )
        # 初始化透明VAE组件
        self.transparent_encoder = TransparentVAEEncoder(encoder_path).to(self.device, dtype=torch.float16)
        self.transparent_decoder = TransparentVAEDecoder(decoder_path).to(self.device, dtype=torch.float16)

    def _init_pipeline(self):
        """初始化KDiffusion pipeline(论文中兼容的采样方式)"""
        self.pipeline = KDiffusionStableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=None  # 使用A1111风格采样
        )

    def _encode_prompt(self, prompt_pos: str, prompt_neg: str):
        """编码正负prompt为模型输入的embedding"""
        with torch.inference_mode():
            positive_cond, positive_pooler = self.pipeline.encode_cropped_prompt_77tokens(prompt_pos)
            negative_cond, negative_pooler = self.pipeline.encode_cropped_prompt_77tokens(prompt_neg)
            # 转移到设备并保持float16精度
            return (
                positive_cond.to(self.device, dtype=torch.float16),
                positive_pooler.to(self.device, dtype=torch.float16), # type: ignore
                negative_cond.to(self.device, dtype=torch.float16),
                negative_pooler.to(self.device, dtype=torch.float16) # type: ignore
            )

    def _generate_latents(self, width: int, height: int, prompt_embeds: tuple, num_steps: int = 25, guidance_scale: float = 7.0, seed: int | None = None):
        """生成潜变量（通用潜变量生成逻辑,供各功能调用)"""
        positive_cond, positive_pooler, negative_cond, negative_pooler = prompt_embeds
        seed = seed if seed is not None else random.randint(0, 1000000)
        rng = torch.Generator(device=self.device).manual_seed(seed)

        # 初始化潜变量（BCHW,SDXL潜变量尺寸为原图的1/8)
        latent_shape = (1, 4, height // 8, width // 8)
        initial_latent = torch.zeros(latent_shape, dtype=torch.float16, device=self.device)

        # 运行KDiffusion采样生成潜变量
        with torch.inference_mode():
            latents_out = self.pipeline(
                initial_latent=initial_latent,
                strength=1.0,
                num_inference_steps=num_steps,
                batch_size=1,
                prompt_embeds=positive_cond,
                negative_prompt_embeds=negative_cond,
                pooled_prompt_embeds=positive_pooler,
                negative_pooled_prompt_embeds=negative_pooler,
                generator=rng,
                guidance_scale=guidance_scale
            )
        return latents_out, seed

    def gen_single_transparent(self, width: int = 1024, height: int = 1024, prompt_pos: str = "", prompt_neg: str = "face asymmetry, deformed, blurry", seed: int | None = None) -> Image.Image:
        """
        功能1:单一透明图像生成
        返回:单张透明图像
        """
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8 (SDXL latent scale)")
        
        # 编码prompt + 生成潜变量
        prompt_embeds = self._encode_prompt(prompt_pos, prompt_neg)
        latents_out, used_seed = self._generate_latents(width, height, prompt_embeds, seed=seed)

        # 透明VAE解码为RGBA图像
        with torch.inference_mode():
            # VAE解码需归一化（SDXL VAE默认缩放因子0.18215)
            latents_norm = latents_out.to(dtype=torch.float16, device=self.device) / 0.18215
            result_list = self.transparent_decoder(self.vae, latents_norm)
            rgba_image = Image.fromarray(result_list[0]).convert("RGBA")  # 转为RGBA确保透明度保留

        print(f"Single transparent image generated (seed: {used_seed})")
        return rgba_image

    def gen_multi_layers_joint(self, width: int = 1024, height: int = 1024, prompts: list = [], prompt_neg: str = "face asymmetry, deformed, blurry", seed: int | None = None) -> list[Image.Image]:
        """
        功能2:多层透明图像联合生成(论文3.3节)
        输入:prompts列表(每个元素对应一个图层的prompt,如[前景prompt, 背景prompt])
        返回:多个透明图层(RGBA格式列表)
        """
        if len(prompts) < 2:
            raise ValueError("Joint multi-layer generation requires at least 2 prompts (e.g., foreground + background)")
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be multiples of 8")
        
        layers = []
        used_seed = seed if seed is not None else random.randint(0, 1000000)
        print(f"Generating {len(prompts)} joint layers (base seed: {used_seed})")

        for i, prompt_pos in enumerate(prompts):
            # 为每个图层分配不同子种子（确保图层一致性)
            layer_seed = used_seed + i
            # 生成单个图层（复用单一透明生成逻辑,通过种子关联确保联合一致性)
            layer = self.gen_single_transparent(
                width=width, height=height,
                prompt_pos=prompt_pos, prompt_neg=prompt_neg,
                seed=layer_seed
            )
            layers.append(layer)
        return layers

    def gen_conditional_layer(self, ref_layer: Image.Image, prompt_target: str, is_foreground_condition: bool = True, width: int = 1024, height: int = 1024, prompt_neg: str = "face asymmetry, deformed, blurry", seed: int | None = None) -> Image.Image:
        """
        功能3:条件性图层生成
        输入:
            - ref_layer: 参考图层(RGBA格式,前景/背景参考)
            - prompt_target: 目标图层prompt(如参考前景→生成背景)
            - is_foreground_condition: True=前景条件生成背景;False=背景条件生成前景
        返回:目标透明图层(RGBA格式)
        """
        if ref_layer.mode != "RGBA":
            raise ValueError("Reference layer must be RGBA format (transparent)")
        
        # 1. 参考图层编码为潜变量（作为条件输入)
        with torch.inference_mode():
            # 参考图层转为张量（HWC→CHW,归一化到[-1,1])
            ref_np = np.array(ref_layer).astype(np.float32) / 255.0
            ref_rgb = ref_np[..., :3] * ref_np[..., 3:4]  # 预乘alpha（论文4.1节定义)
            ref_tensor = torch.from_numpy(ref_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device, dtype=torch.float16)
            ref_tensor = (ref_tensor * 2.0) - 1.0  # SDXL输入归一化（[0,1]→[-1,1])
            
            # 参考图层编码为潜变量
            ref_latent = self.vae.encode(ref_tensor).latent_dist.sample() * 0.18215  # VAE编码缩放

        # 2. 目标图层prompt编码
        prompt_embeds = self._encode_prompt(prompt_target, prompt_neg)
        positive_cond, positive_pooler, negative_cond, negative_pooler = prompt_embeds

        # 3. 条件生成:将参考潜变量与目标潜变量合并优化（论文共享注意力机制)
        seed = seed if seed is not None else random.randint(0, 1000000)
        rng = torch.Generator(device=self.device).manual_seed(seed)
        latent_shape = (1, 4, height // 8, width // 8)
        initial_latent = torch.zeros(latent_shape, dtype=torch.float16, device=self.device)

        with torch.inference_mode():
            # 条件潜变量合并（参考潜变量作为额外条件注入)
            if is_foreground_condition:
                # 前景条件:参考潜变量（前景)与初始潜变量（背景)拼接
                cond_latent = torch.cat([ref_latent, initial_latent], dim=1)  # 通道维度拼接
            else:
                # 背景条件:初始潜变量（前景)与参考潜变量（背景)拼接
                cond_latent = torch.cat([initial_latent, ref_latent], dim=1)

            # 运行条件采样（调整UNet输入为条件潜变量)
            latents_out = self.pipeline(
                initial_latent=cond_latent[:, :4, :, :],  # 取前4通道作为UNet输入（兼容原始结构)
                strength=1.0,
                num_inference_steps=25,
                batch_size=1,
                prompt_embeds=positive_cond,
                negative_prompt_embeds=negative_cond,
                pooled_prompt_embeds=positive_pooler,
                negative_pooled_prompt_embeds=negative_pooler,
                generator=rng,
                guidance_scale=7.5,  # 条件生成适当提高引导权重
                extra_cond_latent=cond_latent[:, 4:, :, :]  # 额外条件潜变量（参考图层)
            )

            # 解码为目标透明图层
            latents_norm = latents_out / 0.18215
            result_list = self.transparent_decoder(self.vae, latents_norm)
            target_layer = Image.fromarray(result_list[0]).convert("RGBA")

        print(f"Conditional layer generated (seed: {seed}, {'foreground→background' if is_foreground_condition else 'background→foreground'})")
        return target_layer

    def gen_iterative_layers(self, base_prompt: str, layer_prompts: list, width: int = 1024, height: int = 1024, prompt_neg: str = "face asymmetry, deformed, blurry", seed: int | None = None) -> list[Image.Image]:
        """
        功能4:迭代式多层生成(论文4.3节)
        输入:
            - base_prompt: 基础图层prompt(如“空房间”)
            - layer_prompts: 迭代图层prompt列表(如["桌子", "猫", "桌上植物"])
        返回:所有迭代生成的图层（基础图层 + 迭代图层,均为RGBA格式)
        """
        # 1. 生成基础图层（作为初始背景)
        base_layer = self.gen_single_transparent(
            width=width, height=height,
            prompt_pos=base_prompt, prompt_neg=prompt_neg,
            seed=seed
        )
        all_layers = [base_layer]
        used_seed = seed if seed is not None else random.randint(0, 1000000)

        # 2. 迭代生成后续图层（每次以“当前所有图层混合结果”为背景条件)
        for i, layer_prompt in enumerate(layer_prompts):
            # 混合当前所有图层为临时背景（RGBA混合)
            temp_bg = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for layer in all_layers:
                temp_bg = Image.alpha_composite(temp_bg, layer)
            
            # 以混合背景为条件,生成当前目标图层（背景→前景条件)
            current_layer = self.gen_conditional_layer(
                ref_layer=temp_bg,
                prompt_target=layer_prompt,
                is_foreground_condition=False,  # 背景条件生成前景
                width=width, height=height,
                prompt_neg=prompt_neg,
                seed=used_seed + i + 1  # 子种子确保一致性
            )
            all_layers.append(current_layer)
            print(f"Iterative layer {i+1}/{len(layer_prompts)} generated: {layer_prompt}")
        
        return all_layers

    @staticmethod
    def blend_layers(layers: list[Image.Image], width: int = 1024, height: int = 1024) -> Image.Image:
        """辅助功能:混合多个透明图层为单张RGB图像(论文图4/7展示用)"""
        blended = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for layer in layers:
            blended = Image.alpha_composite(blended, layer)
        return blended.convert("RGB")  # 转为RGB（透明区域变黑)


# ------------------- 功能调用示例 -------------------
if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("./static", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # 1. 初始化生成器
    generator = TransparentImageGenerator()

    # 2. 功能1:生成单张透明图像（如“透明玻璃杯子”)
    single_img = generator.gen_single_transparent(
        prompt_pos="a transparent glass cup, realistic texture, studio lighting",
        seed=12345
    )
    single_img.save("./static/single_glass_cup.png")
    print("Single transparent image saved to ./static/single_glass_cup.png")

    # 3. 功能2:联合生成2个图层（前景“猫” + 背景“花园”)
    multi_layers = generator.gen_multi_layers_joint(
        prompts=[
            "a cute cat, transparent fur details, sitting",  # 前景图层
            "a small garden with grass and flowers, sunny day"  # 背景图层
        ],
        seed=67890
    )
    multi_layers[0].save("./static/multi_foreground_cat.png")  # 前景猫
    multi_layers[1].save("./static/multi_background_garden.png")  # 背景花园
    # 混合图层并保存
    blended_multi = generator.blend_layers(multi_layers)
    blended_multi.save("./static/multi_blended_cat_garden.png")
    print("Joint multi-layers saved to ./static (foreground/background/blended)")

    # 4. 功能3:条件生成（以“透明杯子”为前景,生成“厨房台面”背景)
    # 先加载参考前景图层（或用功能1生成的single_glass_cup.png)
    ref_foreground = Image.open("./static/single_glass_cup.png").convert("RGBA")
    conditional_bg = generator.gen_conditional_layer(
        ref_layer=ref_foreground,
        prompt_target="a kitchen countertop with wooden texture, minimal style",
        is_foreground_condition=True,  # 前景→背景条件
        seed=54321
    )
    conditional_bg.save("./static/conditional_background_kitchen.png")
    # 混合前景与条件背景
    blended_conditional = generator.blend_layers([ref_foreground, conditional_bg])
    blended_conditional.save("./static/conditional_blended_cup_kitchen.png")
    print("Conditional background saved to ./static (background/blended)")

    # 5. 功能4:迭代生成（基础“空房间”→迭代“桌子”→“笔记本电脑”→“咖啡杯”)
    iterative_layers = generator.gen_iterative_layers(
        base_prompt="an empty room with white walls and wooden floor, soft light",
        layer_prompts=[
            "a wooden desk, simple design",  # 第1层:桌子
            "a black laptop on the desk, open",  # 第2层:笔记本电脑
            "a small coffee cup on the desk, next to laptop"  # 第3层:咖啡杯
        ],
        seed=98765
    )
    # 保存所有迭代图层和最终混合结果
    for i, layer in enumerate(iterative_layers):
        layer_name = ["base_room", "desk", "laptop", "coffee_cup"][i]
        layer.save(f"./static/iterative_{layer_name}.png")
    blended_iterative = generator.blend_layers(iterative_layers)
    blended_iterative.save("./static/iterative_blended_final.png")
    print("Iterative layers saved to ./static (all layers + final blended)")