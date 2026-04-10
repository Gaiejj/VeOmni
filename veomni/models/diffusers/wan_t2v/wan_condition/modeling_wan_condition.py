from __future__ import annotations

import math
from typing import Any

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.video_processor import VideoProcessor
from torchvision.transforms import InterpolationMode, functional
from transformers import AutoTokenizer, PreTrainedModel, UMT5EncoderModel

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from .....utils.device import get_device_type
from .configuration_wan_condition import WanTransformer3DConditionModelConfig


logger = logging.get_logger(__name__)


# T2V only
class WanTransformer3DConditionModel(PreTrainedModel):
    config_class = WanTransformer3DConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: WanTransformer3DConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.video_processor = None
        self.negative_prompt_embeds = None
        self._timesteps_ready = False
        self.meta_init = meta_init
        self.seed = config.seed
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed(self.seed + get_parallel_state().dp_rank)
        self._load_components()

    @property
    def _execution_device(self):
        return self.vae.device

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Wan condition components from {base}.")
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
        )
        if self.meta_init:
            self.vae = AutoencoderKLWan.from_config(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        else:
            self.vae = AutoencoderKLWan.from_pretrained(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base,
            subfolder=self.config.scheduler_subfolder,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.scale_factor_spatial)
        self._prepare_negative_prompt_embeds()
        if self.meta_init:
            del self.text_encoder

    @torch.no_grad()
    def _prepare_negative_prompt_embeds(self):
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=[self.config.cfg_negative_prompt],
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.negative_prompt_embeds = prompt_embeds[0].unsqueeze(0)

    def _encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        # resize video to max size
        height, width = video.shape[-2:]

        size = min(self.config.video_max_size, min(width, height))
        video = functional.resize(video, size, interpolation=InterpolationMode.BICUBIC).float().clamp(0, 255)
        video = self.video_processor.preprocess_video(video)
        video = video.to(device=self.vae.device, dtype=self.vae.dtype)

        # save mean & logvar
        posterior: DiagonalGaussianDistribution = self.vae.encode(video).latent_dist

        return posterior.parameters

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        return (latents - latents_mean) / latents_std

    @torch.no_grad()
    def _get_t5_prompt_embeds(self, **kwargs):
        return WanPipeline._get_t5_prompt_embeds(self, **kwargs)

    @torch.no_grad()
    def get_condition(self, inputs, videos, **kwargs) -> dict[str, Any]:
        """
        inputs: list[str], a list of samples of prompts
        videos: list[list[torch.Tensor]] a list of samples of videos
        """
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=inputs,
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )  # bs, seqlen, dim
        context_list = [u.unsqueeze(0) for u in prompt_embeds]

        latents_list: list[list[torch.Tensor]] = []
        for sample_videos in videos:
            assert len(sample_videos) == 1, "Only one video per sample is supported for T2V"
            latents_list.append(self._encode_video_to_latents(sample_videos[0]))  # 1, c, f, h, w

        return {"latents": latents_list, "context": context_list}

    def _sample_timesteps(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample training timesteps according to the configured strategy.

        Args:
            batch_size: Number of timesteps to sample.
            device: Target device for the returned tensor.
            dtype: Target dtype for the returned tensor.

        Returns:
            Tensor of shape ``(batch_size,)`` with sampled timesteps.
        """
        sampling = self.config.timestep_sampling
        if sampling == "uniform":
            timestep_ids = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (batch_size,),
                device=self.generator.device,
                generator=self.generator,
            )
            return self.scheduler.timesteps[timestep_ids].to(device=device, dtype=dtype)
        elif sampling == "logit_normal":
            u = torch.normal(
                mean=self.config.logit_normal_mean,
                std=self.config.logit_normal_std,
                size=(batch_size,),
                device=self.generator.device,
                generator=self.generator,
            )
            u = torch.sigmoid(u)  # Map to [0, 1]
            # Apply flow-matching shift
            u = self.config.shift * u / (1 + (self.config.shift - 1) * u)
            timestep = u * self.config.num_train_timesteps
            return timestep.to(device=device, dtype=dtype)
        elif sampling == "cosmap":
            u = torch.rand(batch_size, device=self.generator.device, generator=self.generator)
            u = 1 - torch.cos(u * math.pi / 2)  # Cosine mapping, concentrates in middle
            # Apply flow-matching shift
            u = self.config.shift * u / (1 + (self.config.shift - 1) * u)
            timestep = u * self.config.num_train_timesteps
            return timestep.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown timestep_sampling strategy: {sampling}")

    def _compute_loss_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Compute per-sample loss weights based on the sampled timesteps.

        Args:
            timestep: Tensor of shape ``(batch_size,)`` with timestep values.

        Returns:
            Tensor of shape ``(batch_size,)`` with loss weights.
        """
        weighting = self.config.loss_weighting
        if weighting == "none":
            return torch.ones_like(timestep)
        elif weighting == "min_snr":
            # For rectified flow: SNR = (1 - sigma)^2 / sigma^2
            sigma = timestep / self.config.num_train_timesteps
            snr = ((1 - sigma) / sigma.clamp(min=1e-6)) ** 2
            min_snr_gamma = 5.0
            return torch.clamp(snr, max=min_snr_gamma) / snr.clamp(min=1e-6)
        elif weighting == "cosmap":
            sigma = timestep / self.config.num_train_timesteps
            weight = 1.0 / (1.0 - sigma + 1e-3)
            return torch.clamp(weight, max=10.0)
        else:
            raise ValueError(f"Unknown loss_weighting strategy: {weighting}")

    def process_condition(self, latents: list[torch.Tensor], context: list[torch.Tensor]) -> dict[str, Any]:
        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=latents[0].device)
            self._timesteps_ready = True

        packed_conditions: dict[str, list[torch.Tensor]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "training_target": [],
            "latents": [],
            "loss_weight": [],
        }
        for sample_latents, sample_context in zip(latents, context):
            latents = DiagonalGaussianDistribution(sample_latents).mode()
            latents = self._normalize_latents(latents).to(self.generator.device)
            noise = torch.randn(  # TODO: use randn_like(generator=self.generator) when updating to torch 2.10.0
                latents.shape, dtype=latents.dtype, device=self.generator.device, generator=self.generator
            ).to(self.generator.device)

            timestep = self._sample_timesteps(latents.shape[0], latents.device, latents.dtype)

            # Compute noisy latents: for non-uniform sampling the timestep may be
            # continuous, so we compute the linear interpolation directly instead
            # of relying on scheduler.scale_noise which expects discrete timesteps.
            if self.config.timestep_sampling == "uniform":
                noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
            else:
                sigma = (timestep / self.config.num_train_timesteps).view(-1, 1, 1, 1, 1)
                noisy_latents = (1 - sigma) * latents + sigma * noise

            training_target = noise - latents
            loss_weight = self._compute_loss_weight(timestep)

            use_negative_context = (
                torch.rand((), device=self.generator.device, generator=self.generator) < self.config.cfg_negative_prob
            )
            if use_negative_context:
                sample_context = self.negative_prompt_embeds.to(device=latents.device, dtype=sample_context.dtype)
            else:
                sample_context = sample_context.to(latents.device)

            packed_conditions["hidden_states"].append(noisy_latents)
            packed_conditions["timestep"].append(timestep)
            packed_conditions["encoder_hidden_states"].append(sample_context)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(latents)
            packed_conditions["loss_weight"].append(loss_weight)

        return packed_conditions
