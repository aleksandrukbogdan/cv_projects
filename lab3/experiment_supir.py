"""Тот же опыт, что в experiment.py, но модель — SUPIR, семплер — EDM с маской.

Базовый прогон x4-upscaler не затрагивается: картинки пишутся в results_supir/.
Запускать из окружения, где установлены зависимости SUPIR, с корнем репозитория
SUPIR в PYTHONPATH. LLaVA не грузится: подпись фиксированная, как у baseline.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything

from degrade import HR_SIZE, make_degradations, make_masks, prepare_hr
from sampler import NEGATIVE_PROMPT, PROMPT
from supir_mask import MaskedRestoreEDMSampler


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
MASK_ORDER = ("zeros", "combined", "ones")
DEG_ORDER = ("bicubic", "unknown", "realesrgan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supir-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", nargs="*", default=["cat", "bridge", "mountain"])
    parser.add_argument("--degradations", nargs="*", default=list(DEG_ORDER))
    parser.add_argument("--out", type=Path, default=ROOT / "results_supir")
    args = parser.parse_args()

    device = "cuda"
    model = load_model(args.supir_root, args.weights, device)
    images = load_images(args.images)
    out = args.out
    for folder in ("hr", "lq", "outputs"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    for name, hr in images.items():
        hr.save(out / "hr" / f"{name}.png")
        degraded = make_degradations(hr, seed=args.seed)
        for deg_name in args.degradations:
            lq = degraded[deg_name]
            lq.save(out / "lq" / f"{name}_{deg_name}.png")
            masks = make_masks(lq.height, lq.width)
            image = to_supir_input(lq, device)
            for mask_name in MASK_ORDER:
                print(f"{name} / {deg_name} / {mask_name}", flush=True)
                prediction = sample_masked(
                    model,
                    image,
                    masks[mask_name],
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=args.seed,
                    device=device,
                )
                if prediction.size != (HR_SIZE, HR_SIZE):
                    prediction = prediction.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
                prediction.save(out / "outputs" / f"{name}_{deg_name}_{mask_name}.png")
                print(f"  сохранено {prediction.size}", flush=True)
            torch.cuda.empty_cache()
    print(f"Готово: {out}")


def load_model(supir_root, weights, device):
    from SUPIR.util import create_SUPIR_model, convert_dtype

    config_path = write_local_config(supir_root, weights)
    model = create_SUPIR_model(str(config_path), SUPIR_sign="Q")
    model.ae_dtype = convert_dtype("bf16")
    model.model.dtype = convert_dtype("fp16")
    model.half()
    model = model.to(device)
    model.eval()
    return model


def write_local_config(supir_root, weights):
    source = Path(supir_root) / "options" / "SUPIR_v0.yaml"
    config = OmegaConf.load(source)
    config.SDXL_CKPT = str(Path(weights) / "sd_xl_base_1.0_0.9vae.safetensors")
    config.SUPIR_CKPT_Q = str(Path(weights) / "SUPIR-v0Q.ckpt")
    config.SUPIR_CKPT_F = config.SUPIR_CKPT_Q
    config.SUPIR_CKPT = None
    config.model.params.network_config.params.spatial_transformer_attn_type = "softmax"
    config.model.params.control_stage_config.params.spatial_transformer_attn_type = "softmax"
    config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"
    path = Path(supir_root) / "options" / "SUPIR_v0_local.yaml"
    OmegaConf.save(config, path)
    return path


def load_images(names):
    images = {}
    for name in names:
        path = IMAGE_DIR / f"{name}.jpg"
        if not path.exists():
            raise SystemExit(f"Нет кадра {path}. Его кладёт baseline-прогон.")
        images[name] = prepare_hr(Image.open(path))
    return images


def to_supir_input(lq, device):
    """Вход SUPIR — уже растянутый до 512 испорченный кадр. Это и есть bicubic-запас."""
    enlarged = lq.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(enlarged).astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device)


@torch.no_grad()
def sample_masked(model, image, mask, steps, guidance, seed, device):
    seed_everything(seed)
    known = model.encode_first_stage(image)
    control = model.encode_first_stage_with_denoise(image, use_sample=False)
    positive, negative = model.prepare_condition(control, [PROMPT], "", NEGATIVE_PROMPT, 1)
    mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(device=device)[None, None]
    if mask_t.shape[-2:] != known.shape[-2:]:
        mask_t = torch.nn.functional.interpolate(mask_t.float(), size=known.shape[-2:], mode="nearest")

    guider = model.sampler_config.params.guider_config
    guider.params.scale = guidance
    guider.params.scale_min = guidance
    sampler = MaskedRestoreEDMSampler(
        mask=mask_t,
        known_latent=known,
        num_steps=steps,
        s_churn=0.0,
        s_noise=1.01,
        restore_cfg=-1.0,
        discretization_config=model.sampler_config.params.discretization_config,
        guider_config=guider,
        verbose=False,
        device=device,
    )

    def denoiser(latent_input, sigma, condition, control_scale):
        return model.denoiser(model.model, latent_input, sigma, condition, control_scale)

    noise = torch.randn_like(known)
    sampled = sampler(
        denoiser,
        noise,
        cond=positive,
        uc=negative,
        x_center=known,
        control_scale=1.0,
    )
    decoded = model.decode_first_stage(sampled)
    frame = decoded[0].float().clamp(-1, 1)
    array = ((frame.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
