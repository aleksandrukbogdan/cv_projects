"""Маска в семплере SUPIR.

SUPIR чистит латент шагами EDM: состояние — это чистый код плюс шум уровня sigma.
После шага сети вне маски подставляется код bicubic-увеличения с тем же уровнем шума.
Внутри маски остаётся шаг модели. На последнем шаге sigma = 0, снаружи лежит чистый bicubic.
"""

import torch
import torch.nn.functional as F
from sgm.modules.diffusionmodules.sampling import RestoreEDMSampler


class MaskedRestoreEDMSampler(RestoreEDMSampler):
    def __init__(self, mask, known_latent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask
        self.known_latent = known_latent

    def sampler_step(
        self,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
        gamma=0.0,
        x_center=None,
        eps_noise=None,
        control_scale=1.0,
        use_linear_control_scale=False,
        control_scale_start=0.0,
    ):
        updated = super().sampler_step(
            sigma,
            next_sigma,
            denoiser,
            x,
            cond,
            uc,
            gamma,
            x_center,
            eps_noise,
            control_scale,
            use_linear_control_scale,
            control_scale_start,
        )
        return self._apply_mask(updated, next_sigma)

    def _apply_mask(self, latent, next_sigma):
        level = float(next_sigma.detach().flatten()[0])
        if level <= 0:
            known = self.known_latent
        else:
            scale = next_sigma
            while scale.ndim < latent.ndim:
                scale = scale.unsqueeze(-1)
            noise = torch.randn(latent.shape, device=latent.device, dtype=torch.float32)
            known = self.known_latent.float() + noise * scale.float()
        mask = self.mask
        if mask.shape[-2:] != latent.shape[-2:]:
            mask = F.interpolate(mask.float(), size=latent.shape[-2:], mode="nearest")
        mask = mask.to(device=latent.device, dtype=latent.dtype)
        return mask * latent + (1.0 - mask) * known.to(dtype=latent.dtype)
