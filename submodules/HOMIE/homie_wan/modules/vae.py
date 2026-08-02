# VAE wrapper around the HuggingFace diffusers AutoencoderKLWan.
#
# Mirrors the numeric behaviour of the NPU AEModel(wan_video_vae) path:
#   * from_pretrained on the `vae` subfolder of Wan2.1-T2V-14B-Diffusers,
#   * encode: latent_dist.mode()  (config do_sample:false) then normalize with
#     channel_specified_shift_scale: (x - latents_mean) / latents_std,
#   * decode: denormalize with x / (1/latents_std) + latents_mean, then .decode().sample.
# See homie_npu diffusers_ae_model.py + wan_pipeline.py:246-258.
import os

import torch


class HomieWanVAE:
    def __init__(self, ckpt_dir, vae_subfolder="vae",
                 norm_mode="channel_specified_shift_scale",
                 dtype=torch.bfloat16, device="cuda"):
        from diffusers import AutoencoderKLWan

        self.device = device
        self.dtype = dtype
        self.norm_mode = norm_mode
        self.model = AutoencoderKLWan.from_pretrained(
            os.path.join(ckpt_dir, vae_subfolder), torch_dtype=dtype
        )
        self.model.to(device).eval()
        self.model.requires_grad_(False)

        cfg = self.model.config
        self.z_dim = cfg.z_dim
        # per-channel latent statistics, shaped [1, z_dim, 1, 1, 1]
        self._mean = torch.tensor(cfg.latents_mean).view(1, self.z_dim, 1, 1, 1)
        self._std = torch.tensor(cfg.latents_std).view(1, self.z_dim, 1, 1, 1)

    def _mean_std(self, ref):
        return self._mean.to(ref.device, ref.dtype), self._std.to(ref.device, ref.dtype)

    @torch.no_grad()
    def encode(self, x):
        """x: [B, 3, F, H, W] in [-1, 1]. Returns normalized latents [B, z_dim, f, h, w]."""
        x = x.to(self.device, self.dtype)
        out = self.model.encode(x, return_dict=True).latent_dist.mode()
        if self.norm_mode == "channel_specified_shift_scale":
            mean, std = self._mean_std(out)
            out = (out - mean) / std
        return out

    @torch.no_grad()
    def decode(self, latents):
        """latents: normalized [B, z_dim, f, h, w]. Returns video [B, 3, F, H, W] in [-1, 1]."""
        latents = latents.to(self.device, self.dtype)
        if self.norm_mode == "channel_specified_shift_scale":
            mean, std = self._mean_std(latents)
            # inverse of encode: x / (1/std) + mean  == x*std + mean.
            latents = latents / (1.0 / std) + mean
        return self.model.decode(latents, return_dict=True).sample
