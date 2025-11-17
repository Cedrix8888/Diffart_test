import torch
from tqdm.auto import trange
from typing import Optional, Union, List
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import  StableDiffusionXLPipelineOutput
from diffusers.utils.torch_utils import randn_tensor


# DPM-Solver++ (2M) Sampling Algorithm
@torch.no_grad()
def sample_dpmpp_2m(model, x: torch.Tensor, sigmas: torch.Tensor, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

class KModel:
    def __init__(self, unet, timesteps=1000, linear_start=0.00085, linear_end=0.012):
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0).clone().detach()

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0
        self.unet = unet

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.0):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def __call__(self, x, sigma, **extra_args):
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data ** 2) ** 0.5
        t = self.timestep(sigma)
        cfg_scale = extra_args['cfg_scale']
        eps_positive = self.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
        eps_negative = self.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
        noise_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        return x - noise_pred * sigma[:, None, None, None]


class KDiffusionStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_model = KModel(unet=kwargs['unet'])
           
    @torch.inference_mode()
    def encode_cropped_prompt_77tokens(self, prompt: str):
        device = self.unet.device
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        prompt_embeds_list = []
        pooled_prompt_embeds = None
        
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # Only last pooler_output is needed
            pooled_prompt_embeds = prompt_embeds.pooler_output
            if pooled_prompt_embeds is None:
                pooled_prompt_embeds = torch.zeros(
                    (1, text_encoder.config.hidden_size),
                    device=device,
                    dtype=self.unet.dtype
                )

            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        return prompt_embeds, pooled_prompt_embeds

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent: Optional[torch.Tensor] = None,
            strength: float = 1.0,
            height: int = 1024,
            width: int = 1024,
            num_inference_steps: int = 25,
            guidance_scale: float = 5.0,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        device = self.unet.device
        
        # Sigmas
        sigmas = self.k_model.get_sigmas_karras(int(num_inference_steps/strength))
        sigmas = sigmas[-(num_inference_steps + 1):].to(device)

        # Initial latents
        if initial_latent is None:
            batch_size = batch_size or 1
            initial_latent = torch.randn(
                (batch_size, self.unet.in_channels, height // 8, width // 8),
                device=device,
                dtype=self.unet.dtype
            )
        _, C, H, W = initial_latent.shape
        noise = randn_tensor((batch_size, C, H, W), generator=generator, device=device, dtype=self.unet.dtype)
        latents = initial_latent.to(noise) + noise * sigmas[0].to(noise)

        # Shape
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        add_time_ids = list((height, width) + (0, 0) + (height, width))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.unet.dtype)
        add_neg_time_ids = add_time_ids.clone()

        # Prompt
        if  prompt_embeds is None or pooled_prompt_embeds is None:            
            prompt = ""
            prompt_embeds, pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(prompt)
            assert prompt_embeds is not None, "Failed to generate prompt_embeds"
            assert pooled_prompt_embeds is not None, "Failed to generate pooled_prompt_embeds"
        if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None:
            negative_prompt = ""
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_cropped_prompt_77tokens(negative_prompt)
            assert negative_prompt_embeds is not None, "Failed to generate negative_prompt_embeds"
            assert negative_pooled_prompt_embeds is not None, "Failed to generate negative_pooled_prompt_embeds"

        # Batch
        latents_in = latents.to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(device)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1).to(device)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1).to(device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1).to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1).to(device)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1).to(device)

        # Feeds
        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
            )
        )

        # Result
        latents_out = sample_dpmpp_2m(self.k_model, latents_in, sigmas, extra_args=sampler_kwargs, disable=False)
        return latents_out