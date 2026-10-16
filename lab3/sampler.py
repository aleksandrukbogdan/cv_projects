"""Сэмплинг Stable Diffusion x4 Upscaler с маской (SR-inpainting).

Модель — латентная диффузия. На каждом шаге UNet предлагает, каким должен
стать латент. Идея RePaint: вне маски это предложение заменяется зашумлённым
латентом bicubic-увеличения. Внутри маски остаётся предложение модели.
К последнему шагу вне маски лежит чистый латент bicubic, внутри — восстановление.
"""

import numpy as np
import torch
import torch.nn.functional as F


PROMPT = "sharp, detailed, high resolution photograph"
NEGATIVE_PROMPT = "blurry, noisy, jpeg artifacts, low quality, oversaturated"


@torch.no_grad()
def super_resolve_masked(
    pipe,
    lq,
    mask,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=25,
    guidance_scale=7.5,
    noise_level=20,
    seed=0,
):
    if mask.ndim != 2:
        raise ValueError("Маска должна быть двумерным массивом HxW")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Маска бинарная: значения только 0 и 1")

    device = pipe._execution_device
    compute_dtype = next(pipe.unet.parameters()).dtype
    do_cfg = guidance_scale > 1.0
    generator = torch.Generator(device=device).manual_seed(seed)

    prompt_embeds, negative_embeds = pipe.encode_prompt(
        prompt,
        device,
        1,
        do_cfg,
        negative_prompt,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])

    condition = pipe.image_processor.preprocess(lq)
    condition = condition.to(device=device, dtype=compute_dtype)
    known_latent = _encode_bicubic(pipe, condition, compute_dtype)

    level = torch.tensor([noise_level], dtype=torch.long, device=device)
    condition_noise = torch.randn(
        condition.shape, generator=generator, device=device, dtype=compute_dtype
    )
    condition = pipe.low_res_scheduler.add_noise(condition, condition_noise, level)
    condition = condition.to(dtype=compute_dtype)

    copies = 2 if do_cfg else 1
    condition = condition.repeat(copies, 1, 1, 1)
    level = level.repeat(copies)

    height, width = condition.shape[-2:]
    latents = torch.randn(
        (1, pipe.vae.config.latent_channels, height, width),
        generator=generator,
        device=device,
        dtype=compute_dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra = pipe.prepare_extra_step_kwargs(generator, eta=0.0)
    mask_t = _mask_tensor(mask, latents)

    for timestep in timesteps:
        model_in = torch.cat([latents] * copies) if do_cfg else latents
        model_in = pipe.scheduler.scale_model_input(model_in, timestep)
        model_in = torch.cat([model_in, condition], dim=1)

        noise_pred = pipe.unet(
            model_in,
            timestep,
            encoder_hidden_states=prompt_embeds,
            class_labels=level,
            return_dict=False,
        )[0]
        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        latents = pipe.scheduler.step(
            noise_pred, timestep, latents, **extra, return_dict=False
        )[0]
        latents = _blend_known(pipe.scheduler, latents, known_latent, mask_t, timestep, generator)

    decoded = pipe.vae.decode(
        latents.to(dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0]
    return pipe.image_processor.postprocess(decoded.float(), output_type="pil")[0]


def _encode_bicubic(pipe, condition, compute_dtype):
    """Латент bicubic ×4. Его модель не должна переписывать вне маски."""
    upscaled = F.interpolate(condition.float(), scale_factor=4, mode="bicubic", align_corners=False)
    upscaled = upscaled.clamp(-1, 1).to(dtype=pipe.vae.dtype)
    latent = pipe.vae.encode(upscaled).latent_dist.mean
    latent = latent * pipe.vae.config.scaling_factor
    latent = latent.to(dtype=compute_dtype)
    if latent.shape[-2:] != condition.shape[-2:]:
        latent = F.interpolate(latent, size=condition.shape[-2:], mode="bilinear", align_corners=False)
    return latent


def _mask_tensor(mask, latents):
    tensor = torch.from_numpy(np.ascontiguousarray(mask)).to(device=latents.device, dtype=latents.dtype)
    tensor = tensor[None, None]
    if tensor.shape[-2:] != latents.shape[-2:]:
        tensor = F.interpolate(tensor, size=latents.shape[-2:], mode="nearest")
    return tensor


def _blend_known(scheduler, latents, known_latent, mask, timestep, generator):
    """Вне маски подставляем bicubic-латент с шумом текущего уровня."""
    step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    previous = int(timestep) - step
    if previous >= 0:
        # Шум уровня previous совпадает с тем, на котором scheduler.step оставил латент.
        noise = torch.randn(
            latents.shape, generator=generator, device=latents.device, dtype=torch.float32
        )
        previous_t = torch.tensor([previous], device=latents.device, dtype=torch.long)
        known = scheduler.add_noise(known_latent.float(), noise, previous_t)
    else:
        known = known_latent
    return mask * latents + (1.0 - mask) * known.to(dtype=latents.dtype)
