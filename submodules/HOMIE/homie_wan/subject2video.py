# HOMIE-Wan subject-to-video (r2v) pipeline for CUDA.
#
# Structure follows Phantom `phantom_wan/subject2video.py`; the denoise loop, reference
# latent preparation and post-processing reproduce the NPU WanPipeline
# (homie_npu .../pipeline/wan_pipeline.py) so that the numeric behaviour matches.
#
# Key differences from Phantom's S2V pipeline (all to match HOMIE/NPU):
#   * reference images -> per-image single-frame VAE latents, zero-padded to
#     reference_num along the frame axis (prepare_r2v_image_latents),
#   * qwen (MLLM) feature + reference_id_labels threaded into the DiT,
#   * classifier-free guidance is single text-CFG: uncond + scale*(cond - uncond)
#     (NPU wan_pipeline.py:215-219), not Phantom's dual image/text CFG,
#   * VAE / text-encoder loaded from HF diffusers with channel_specified_shift_scale
#     latent normalization.
import gc
import json
import logging
import os
from contextlib import contextmanager
from functools import partial

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import ImageOps
from safetensors.torch import load_file

from .distributed.fsdp import shard_model
from .modules.model import HomieWanModel
from .modules.t5 import HomieT5
from .modules.vae import HomieWanVAE


def load_sharded_safetensors(model_dir, base_name, device="cpu"):
    """Load a HF-style sharded safetensors checkpoint via its index json.
    Mirrors Phantom's load_custom_sharded_weights (subject2video.py:118-131)."""
    index_path = os.path.join(model_dir, f"{base_name}.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = set(weight_map.values())
    state_dict = {}
    for shard_file in shard_files:
        shard_state = load_file(os.path.join(model_dir, shard_file))
        state_dict.update({k: v.to(device) for k, v in shard_state.items()})
    return state_dict


class HomieWanS2V:
    def __init__(
        self,
        config,
        ckpt_dir,
        homie_ckpt,
        homie_ckpt_basename="Homie_Wan_14B",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        """
        config:      EasyDict from configs (s2v_14B).
        ckpt_dir:    HF Wan2.1-T2V-14B-Diffusers dir (vae/text_encoder/tokenizer subfolders).
        homie_ckpt:  dir containing Homie_Wan_14B*.safetensors + index json.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.reference_num = config.reference_num
        self.patch_size = config.patch_size
        self.sample_neg_prompt = config.sample_neg_prompt

        if use_usp:
            raise NotImplementedError(
                "Context parallel (--ulysses_size > 1) is not supported yet; "
                "see homie_wan/distributed/context_parallel.py."
            )

        shard_fn = partial(shard_model, device_id=device_id)

        # ----- text encoder + tokenizer (UMT5) -----
        self.text_encoder = HomieT5(
            ckpt_dir=ckpt_dir,
            text_encoder_subfolder=config.text_encoder_subfolder,
            tokenizer_subfolder=config.tokenizer_subfolder,
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu") if t5_cpu else self.device,
            shard_fn=shard_fn if t5_fsdp else None,
        )

        # ----- VAE (AutoencoderKLWan) -----
        self.vae = HomieWanVAE(
            ckpt_dir=ckpt_dir,
            vae_subfolder=config.vae_subfolder,
            norm_mode=config.vae_norm_mode,
            dtype=config.vae_dtype,
            device=self.device,
        )
        self.z_dim = self.vae.z_dim
        # AutoencoderKLWan strides: temporal 4, spatial 8 (Wan2.1 VAE).
        self.vae_stride = (4, 8, 8)

        # ----- DiT -----
        logging.info(f"Creating HomieWanModel from {homie_ckpt}")
        self.model = HomieWanModel(
            model_type=config.model_type,
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=config.in_dim,
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            img_dim=config.img_dim,
            out_dim=config.out_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            qk_norm=config.qk_norm,
            qk_norm_type=config.qk_norm_type,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            max_seq_len=config.max_seq_len,
            reference_num=config.reference_num,
            qwen_hidden_size=config.qwen_hidden_size,
        )
        state = load_sharded_safetensors(homie_ckpt, homie_ckpt_basename, device="cpu")
        # Reshape any 0-dim params (e.g. qwen_3dvae_connector.scale) to 1D so they
        # match the model (see qwen_connector.py) — FSDP can't shard scalars.
        for k, v in list(state.items()):
            if v.dim() == 0:
                state[k] = v.reshape(1)
        # strict=True: the model was built with MindSpeed-matching key names.
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logging.warning(f"load_state_dict missing={missing} unexpected={unexpected}")
        self.model = self.model.to(self.param_dtype).eval().requires_grad_(False)

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

    # ---------------------------------------------------------------------------------
    # Reference-latent preparation (port of prepare_r2v_image_latents)
    # ---------------------------------------------------------------------------------
    @staticmethod
    def _pad_image(image, h, w):
        """Aspect-preserving pad to target ratio h/w with white fill, then the caller
        resizes to (w, h). Matches NPU pad_image + resize."""
        target_ratio = h / w
        iw, ih = image.size
        image_ratio = ih / iw
        if image_ratio > target_ratio:
            pad = int((ih / target_ratio - iw) / 2)
            image = ImageOps.expand(image, (pad, 0, pad, 0), fill=(255, 255, 255))
        else:
            pad = int((iw * target_ratio - ih) / 2)
            image = ImageOps.expand(image, (0, pad, 0, pad), fill=(255, 255, 255))
        return image

    def get_ref_latents(self, ref_images, size):
        """ref_images: List[PIL.Image] for one sample. Returns [z_dim, reference_num, h, w]
        (zero-padded along the frame axis to reference_num)."""
        w, h = size  # size is (width, height)
        latent_h, latent_w = h // self.vae_stride[1], w // self.vae_stride[2]

        refs = ref_images[: self.reference_num]
        latents = []
        for img in refs:
            img = self._pad_image(img, h, w).resize((w, h))
            # ToTensor -> [0,1], then Normalize(0.5,0.5) -> [-1,1]; add frame dim.
            t = TF.to_tensor(img).sub_(0.5).div_(0.5)  # [3, h, w]
            t = t.unsqueeze(1)  # [3, 1, h, w]
            lat = self.vae.encode(t.unsqueeze(0))[0]  # [z_dim, 1, latent_h, latent_w]
            latents.append(lat)

        feat = torch.cat(latents, dim=1)  # [z_dim, n_ref, lh, lw]
        d, t, lh, lw = feat.shape
        template = torch.zeros(d, self.reference_num, lh, lw, dtype=feat.dtype, device=feat.device)
        template[:, :t] = feat[:, : self.reference_num]
        return template  # [z_dim, reference_num, latent_h, latent_w]

    # ---------------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_prompt,
        ref_images,
        qwen_feature=None,
        reference_id_labels=None,
        size=(1280, 720),
        frame_num=97,
        shift=3.0,
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=False,
    ):
        """
        input_prompt:        str
        ref_images:          List[PIL.Image]
        qwen_feature:        [1, S, 2048] tensor or None
        reference_id_labels: List[List[str]] e.g. [['a'], ['b'], ['c']]
        size:                (width, height)
        """
        from diffusers import UniPCMultistepScheduler

        w, h = size
        latent_h, latent_w = h // self.vae_stride[1], w // self.vae_stride[2]
        latent_f = (frame_num - 1) // self.vae_stride[0] + 1

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if seed >= 0:
            torch.manual_seed(seed)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed if seed >= 0 else 0)

        # ----- reference latents -----
        ref_latents = self.get_ref_latents(ref_images, size).to(self.device)  # [z,ref,lh,lw]
        ref_latents = ref_latents.unsqueeze(0).to(self.param_dtype)  # [1,z,ref,lh,lw]
        # NOTE: HOMIE (like the NPU WanPipeline) uses single *text* CFG only — the same
        # reference latents are used for the conditional and unconditional passes. There
        # is no separate zeroed-reference (image-CFG) branch as in Phantom's S2V.

        # ----- qwen feature -----
        if qwen_feature is not None:
            qwen_feature = qwen_feature.to(device=self.device, dtype=self.param_dtype)

        # ----- text embeddings -----
        if self.t5_cpu:
            context = self.text_encoder([input_prompt], torch.device("cpu")).to(self.device)
            context_null = self.text_encoder([n_prompt], torch.device("cpu")).to(self.device)
        else:
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
        context = context.to(self.param_dtype)
        context_null = context_null.to(self.param_dtype)

        # ----- initial noise -----
        noise = torch.randn(
            1, self.z_dim, latent_f, latent_h, latent_w,
            dtype=torch.float32, device=self.device, generator=seed_g,
        )

        # ----- scheduler (diffusers UniPCMultistepScheduler, flow matching) -----
        sample_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            solver_order=2,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=self.device)
        timesteps = sample_scheduler.timesteps

        latents = noise

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with torch.amp.autocast("cuda", dtype=self.param_dtype), no_sync():
            for t in tqdm(timesteps, desc="denoising", disable=self.rank != 0):
                latent_model_input = latents.to(self.param_dtype)
                timestep = t.expand(latents.shape[0]).to(self.device).float()

                noise_cond = self.model(
                    latent_model_input, timestep, context,
                    reference=ref_latents, qwen_feature=qwen_feature,
                    reference_id_labels=reference_id_labels,
                )
                noise_uncond = self.model(
                    latent_model_input, timestep, context_null,
                    reference=ref_latents, qwen_feature=qwen_feature,
                    reference_id_labels=reference_id_labels,
                )
                noise_pred = noise_uncond + guide_scale * (noise_cond - noise_uncond)

                latents = sample_scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        # ----- decode -----
        # Each rank decodes the sample it owns (samples are distributed data-parallel,
        # round-robin across ranks in generate.py), so every rank must produce & save
        # its own video — not just rank 0.
        video = self.vae.decode(latents.to(self.vae.dtype))  # [1,3,F,H,W] in [-1,1]
        video = video[0]  # [3, F, H, W]

        del noise, latents, sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        return video
