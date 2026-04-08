# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import inspect
import json
import logging
import math
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import PIL.Image
from PIL import Image
import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.redmedia_image_sr.cfg_parallel import (
    RedMediaImageSRCFGParallelMixin,
)

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.models.redmedia_image_sr.dual_lora_linear import DualLoRALinear, LinearFP8Wrapper, DualLoRAQKVLinear

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import calculate_shift
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    calculate_dimensions,
    retrieve_latents,
    retrieve_timesteps,
)

from vllm_omni.diffusion.models.redmedia_image_sr.redmedia_image_sr_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from copy import deepcopy
from vllm_omni.diffusion.models.redmedia_image_sr.odtsr.generator import Generator
from vllm_omni.diffusion.models.redmedia_image_sr.odtsr.wavelet_color_fix import adain_color_fix, wavelet_color_fix
logger = logging.getLogger(__name__)


# def get_qwen_image_post_process_func(
#     od_config: OmniDiffusionConfig,
# ):
#     model_name = od_config.model
#     if os.path.exists(model_name):
#         model_path = model_name
#     else:
#         model_path = download_weights_from_hf_specific(model_name, None, ["*"])
#     vae_config_path = os.path.join(model_path, "vae/config.json")
#     with open(vae_config_path) as f:
#         vae_config = json.load(f)
#         vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

#     image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

#     # def post_process_func(
#     #     images: torch.Tensor,
#     # ):
#     #     return image_processor.postprocess(images)
#     def pre_process_func(
#         request: OmniDiffusionRequest,
#     ):
#         """Pre-process requests for QwenImageEditPlusPipeline."""
#         for i, prompt in enumerate(request.prompts):
#             multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
#             raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
#             if isinstance(prompt, str):
#                 prompt = OmniTextPrompt(prompt=prompt)
#             if "additional_information" not in prompt:
#                 prompt["additional_information"] = {}

#             # Handle single image or list of images
#             if raw_image is None:
#                 continue

#             if not isinstance(raw_image, list):
#                 raw_image = [raw_image]
#             image = [
#                 PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image | np.ndarray | torch.Tensor, im)
#                 for im in raw_image
#             ]

#             # Calculate dimensions based on first image
#             image_size = image[0].size
#             calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
#             height = request.sampling_params.height or calculated_height
#             width = request.sampling_params.width or calculated_width

#             # Ensure dimensions are multiples of vae_scale_factor * 2
#             multiple_of = vae_scale_factor * 2
#             height = height // multiple_of * multiple_of
#             width = width // multiple_of * multiple_of

#             # Store calculated dimensions in request
#             prompt["additional_information"]["calculated_height"] = calculated_height
#             prompt["additional_information"]["calculated_width"] = calculated_width
#             request.sampling_params.height = height
#             request.sampling_params.width = width

#             # Preprocess images into condition_images (for prompt encoding) and vae_images (for VAE encoding)
#             condition_images = []
#             vae_images = []
#             condition_image_sizes = []
#             vae_image_sizes = []

#             for img in image:
#                 if isinstance(img, torch.Tensor) and len(img.shape) > 1 and img.shape[1] == latent_channels:
#                     # Already a latent tensor
#                     continue

#                 image_width, image_height = img.size
#                 condition_width, condition_height = calculate_dimensions(
#                     CONDITION_IMAGE_SIZE, image_width / image_height
#                 )
#                 vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

#                 condition_image_sizes.append((condition_width, condition_height))
#                 vae_image_sizes.append((vae_width, vae_height))

#                 condition_images.append(image_processor.resize(img, condition_height, condition_width))
#                 vae_images.append(image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

#             # Store preprocessed images in request
#             prompt["additional_information"]["condition_images"] = condition_images
#             prompt["additional_information"]["vae_images"] = vae_images
#             prompt["additional_information"]["condition_image_sizes"] = condition_image_sizes
#             prompt["additional_information"]["vae_image_sizes"] = vae_image_sizes
#             request.prompts[i] = prompt
#         return request

#     return pre_process_func

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

def get_qwen_image_edit_plus_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for QwenImageEditPlusPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)
    latent_channels = vae_config.get("z_dim", 16)

    def pre_process_func(
        request: OmniDiffusionRequest,
    ):
        """Pre-process requests for QwenImageEditPlusPipeline."""
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            # Handle single image or list of images
            if raw_image is None:
                print(f"raw_image is None")
                continue

            if not isinstance(raw_image, list):
                raw_image = [raw_image]
            image = [
                PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image | np.ndarray | torch.Tensor, im)
                for im in raw_image
            ]

            # Calculate dimensions based on first image
            image_size = image[0].size
            calculated_width, calculated_height = calculate_dimensions(VAE_IMAGE_SIZE, image_size[0] / image_size[1])
            height = request.sampling_params.height or calculated_height
            width = request.sampling_params.width or calculated_width

            # Ensure dimensions are multiples of vae_scale_factor * 2
            multiple_of = vae_scale_factor * 2
            height = height // multiple_of * multiple_of
            width = width // multiple_of * multiple_of

            # Store calculated dimensions in request
            prompt["additional_information"]["calculated_height"] = calculated_height
            prompt["additional_information"]["calculated_width"] = calculated_width
            request.sampling_params.height = height
            request.sampling_params.width = width

            # Preprocess images into condition_images (for prompt encoding) and vae_images (for VAE encoding)
            condition_images = []
            vae_images = []
            condition_image_sizes = []
            vae_image_sizes = []

            for img in image:
                if isinstance(img, torch.Tensor) and len(img.shape) > 1 and img.shape[1] == latent_channels:
                    # Already a latent tensor
                    continue

                image_width, image_height = img.size
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)

                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))

                condition_images.append(image_processor.resize(img, condition_height, condition_width))
                vae_images.append(image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

            # Store preprocessed images in request
            prompt["additional_information"]["condition_images"] = condition_images
            prompt["additional_information"]["vae_images"] = vae_images
            prompt["additional_information"]["condition_image_sizes"] = condition_image_sizes
            prompt["additional_information"]["vae_image_sizes"] = vae_image_sizes
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_qwen_image_edit_plus_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-processing function for QwenImageEditPlusPipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class RedMediaImageSRPipeline(nn.Module, RedMediaImageSRCFGParallelMixin, DiffusionPipelineProfilerMixin):
    supports_step_execution: ClassVar[bool] = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.od_config = od_config

        



        """
        print(f"od_config: {od_config}")
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        ).to(self.device)
        self.vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        self.transformer = QwenImageTransformer2DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )

        # self.new_vae = deepcopy(self.vae)
        # self.unfrozen(self.new_vae.encoder, type(self.new_vae.encoder.conv_in))
        self.new_vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)
        

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        self.stage = None

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        # QwenImage latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor * 2
        # )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )
        """

    
    
    def add_custom_dual_lora(self, model, lora_rank):
        patterns = [
            "img_in",
            "img_mod.1",
            # "attn.to_q",
            # "attn.to_k",
            # "attn.to_v",
            "attn.to_qkv", # ?
            "to_out", # ?
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ]
        self.replace_linear_with_duallora(model, patterns, rank=lora_rank, alpha1=0, alpha2=lora_rank, use_fp8=True)

    def replace_linear_with_duallora(self, model, patterns, rank, alpha1, alpha2, use_fp8=False):
        """
        先冻结model所有参数，
        将匹配patterns的nn.Linear替换为DualLoRALinear
        不匹配的替换为 LinearFP8Wrapper
        """
        def to_fp8(tensor):
            # 仅在 use_fp8 时转换
            if use_fp8:
                return tensor.to(dtype=torch.float8_e4m3fn)
            return tensor

        # # 冻结全部参数
        # for p in model.parameters():
        #     p.requires_grad = False

        def _replace_module(parent, name_prefix=""):
            for name, module in list(parent.named_children()):
                full_name = f"{name_prefix}{name}"
                if isinstance(module, ColumnParallelLinear) or isinstance(module, ReplicatedLinear) or isinstance(module, RowParallelLinear) or isinstance(module, QKVParallelLinear):
                    module.weight.data = to_fp8(module.weight.data)
                    if module.bias is not None:
                        module.bias.data = to_fp8(module.bias.data)

                    if any(p in full_name for p in patterns):
                        if "attn.to_qkv" in full_name:
                            # 替换为 DualLoRALinear
                            new_module = DualLoRAQKVLinear(module, rank, alpha1, alpha2)
                            setattr(parent, name, new_module)
                            print(f"[lora] {full_name} -> DualLoRAQKVLinear")
                            
                        else:
                            # 替换为 DualLoRALinear
                            new_module = DualLoRALinear(module, rank, alpha1, alpha2)
                            setattr(parent, name, new_module)
                            print(f"[lora] {full_name} -> DualLoRALinear")
                    else: 
                        # 替换为自动精度转换的 FP8LinearWrapper
                        new_module = LinearFP8Wrapper(module)
                        setattr(parent, name, new_module)
                        print(f"[cast] {full_name} -> LinearFP8Wrapper (fp8={use_fp8})")
                # elif isinstance(module, QKVParallelLinear):
                #     module.weight.data = to_fp8(module.weight.data)
                #     if module.bias is not None:
                #         module.bias.data = to_fp8(module.bias.data)

                #     if any(p in full_name for p in patterns):
                #         # 替换为 DualLoRALinear
                #         new_module = DualLoRAQKVLinear(module, rank, alpha1, alpha2)
                #         setattr(parent, name, new_module)
                #         print(f"[lora] {full_name} -> DualLoRAQKVLinear")
                #     else: 
                #         # 替换为自动精度转换的 FP8LinearWrapper
                #         new_module = LinearFP8Wrapper(module)
                #         setattr(parent, name, new_module)
                #         print(f"[cast] {full_name} -> LinearFP8Wrapper (fp8={use_fp8})")

                else:
                    _replace_module(module, name_prefix=full_name + ".")
        
        _replace_module(model)
    # def unfrozen(self, model, target_cls):
    #     for name, module in model.named_modules():
    #         # time_conv单帧用不到 在不开启find_unused_parameters的情况下会报错
    #         if isinstance(module, target_cls) and 'time_conv' not in name:
    #             for p in module.parameters():
    #                 p.requires_grad = True

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs},
        # but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `prompt_embeds_mask` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        # print(f"attention mask: {txt_tokens.attention_mask}")
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ) -> torch.Tensor:
        # generator=torch.Generator(device="cuda").manual_seed(42)
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _extract_prompts(self, prompts):
        """Extract prompt and negative_prompt from OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    def _prepare_generation_context(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        sigmas,
        guidance_scale,
        num_images_per_prompt,
        generator,
        true_cfg_scale,
        max_sequence_length,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        latents=None,
        attention_kwargs=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """Shared preparation logic for forward() and prepare_encode().

        Validates inputs, encodes prompts, prepares latents, computes timesteps,
        and returns all intermediate values as a dict.
        """
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps,
            sigmas,
            latents.shape[1],
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "latents": latents,
            "img_shapes": img_shapes,
            "timesteps": timesteps,
            "do_true_cfg": do_true_cfg,
            "guidance": guidance,
            "txt_seq_lens": txt_seq_lens,
            "negative_txt_seq_lens": negative_txt_seq_lens,
        }

    def prepare_encode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> "DiffusionRequestState":
        """Populate *state* with encoded prompts, latents, timesteps, and CFG config."""
        sampling = state.sampling
        prompt, negative_prompt = self._extract_prompts(state.prompts or [])

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=sampling.height or self.default_sample_size * self.vae_scale_factor,
            width=sampling.width or self.default_sample_size * self.vae_scale_factor,
            num_inference_steps=sampling.num_inference_steps or 50,
            sigmas=sampling.sigmas,
            guidance_scale=sampling.guidance_scale if sampling.guidance_scale_provided else 1.0,
            num_images_per_prompt=sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1,
            generator=sampling.generator,
            true_cfg_scale=sampling.true_cfg_scale or 4.0,
            max_sequence_length=sampling.max_sequence_length or 512,
            attention_kwargs=kwargs.get("attention_kwargs"),
        )

        # prepare_timesteps() has already materialized request-specific timestep
        # state on self.scheduler, so deepcopy preserves dynamic-shifting state
        # without replaying set_timesteps() on the per-request scheduler.
        # Per-request scheduler (must not share state with self.scheduler)
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        # Populate state from generation context
        state.prompt_embeds = ctx["prompt_embeds"]
        state.prompt_embeds_mask = ctx["prompt_embeds_mask"]
        state.negative_prompt_embeds = ctx["negative_prompt_embeds"]
        state.negative_prompt_embeds_mask = ctx["negative_prompt_embeds_mask"]
        state.latents = ctx["latents"]
        state.timesteps = ctx["timesteps"]
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = ctx["do_true_cfg"]
        state.guidance = ctx["guidance"]
        state.img_shapes = ctx["img_shapes"]
        state.txt_seq_lens = ctx["txt_seq_lens"]
        state.negative_txt_seq_lens = ctx["negative_txt_seq_lens"]
        # QwenImage always normalizes CFG output (matching forward())
        state.sampling.cfg_normalize = True

        return state

    def _build_denoise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list[int] | None,
        do_true_cfg: bool,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        negative_txt_seq_lens: list[int] | None,
        image_latents: torch.Tensor | None = None,
        extra_transformer_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
        """Build positive/negative kwargs and output_slice for one denoise step.

        Returns:
            (positive_kwargs, negative_kwargs, output_slice)
        """
        extra_transformer_kwargs = extra_transformer_kwargs or {}

        # Broadcast timestep to match batch size
        t_for_model = timestep.expand(latents.shape[0]).to(
            device=latents.device,
            dtype=latents.dtype,
        )

        # Concatenate image latents if available (editing pipelines)
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": t_for_model / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            **extra_transformer_kwargs,
        }
        if do_true_cfg:
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": t_for_model / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": negative_txt_seq_lens,
                **extra_transformer_kwargs,
            }
        else:
            negative_kwargs = None

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Unpack, normalize, and VAE-decode latents into a DiffusionOutput."""
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def denoise_step(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """One denoise step: read from *state*, delegate to CFGParallelMixin.

        Reuses ``predict_noise_maybe_with_cfg`` so that CFG-parallel,
        sequential-CFG, and no-CFG paths are handled identically to
        ``diffuse()``.
        """
        if self.interrupt:
            return None

        t = state.current_timestep
        self._current_timestep = t
        self.transformer.do_true_cfg = state.do_true_cfg

        # Normalize timestep to [batch_size] tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=state.latents.device, dtype=state.latents.dtype)

        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=state.latents,
            timestep=t,
            guidance=state.guidance,
            prompt_embeds=state.prompt_embeds,
            prompt_embeds_mask=state.prompt_embeds_mask,
            img_shapes=state.img_shapes,
            txt_seq_lens=state.txt_seq_lens,
            do_true_cfg=state.do_true_cfg,
            negative_prompt_embeds=state.negative_prompt_embeds,
            negative_prompt_embeds_mask=state.negative_prompt_embeds_mask,
            negative_txt_seq_lens=state.negative_txt_seq_lens,
            image_latents=state.sampling.image_latent,
            extra_transformer_kwargs={
                "attention_kwargs": self.attention_kwargs,
                "return_dict": False,
            },
        )

        true_cfg_scale = state.sampling.true_cfg_scale or 4.0
        cfg_normalize = state.sampling.cfg_normalize

        return self.predict_noise_maybe_with_cfg(
            state.do_true_cfg,
            true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            cfg_normalize,
            output_slice,
        )

    def step_scheduler(
        self,
        state: "DiffusionRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One scheduler step: update ``state.latents`` and advance ``step_index``."""
        if self.interrupt:
            return

        t = state.current_timestep
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )

        state.step_index += 1

    def post_decode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents from *state*."""
        self._current_timestep = None

        height = state.sampling.height or self.default_sample_size * self.vae_scale_factor
        width = state.sampling.width or self.default_sample_size * self.vae_scale_factor
        output_type = kwargs.get("output_type", "pil")

        return self._decode_latents(state.latents, height, width, output_type)
    
    def infer(
        self, 
        prompt,
        negative_prompt,
        upsampled_img,
        condition_image, # already paded
        cfg_scale,
        fidelity,
        tiled,
        tile_size,
        tile_stride,
        w_desti,
        h_desti
    ) -> DiffusionOutput:

        with torch.inference_mode():
            res_img = self.model.infer(
                prompt = prompt,
                negative_prompt = negative_prompt,
                condition_image = condition_image,
                cfg_scale = cfg_scale,
                fidelity = fidelity,
                tiled = tiled,
                tile_size = tile_size,
                tile_stride = tile_stride
            )
            # print(f"res_img: {res_img.shape}")
            cropped_image = res_img[:, :, :h_desti, :w_desti]
            # print(f"cropped_image: {cropped_image.shape} {cropped_image.dtype}")
            output_pil = cropped_image
            # output_pil = wavelet_color_fix(target=cropped_image, source=upsampled_img, return_type="Tensor")
            # print(f"output_pil: {output_pil.shape} {output_pil.dtype}")
            # output_pil = output_pil.to(dtype=cropped_image.dtype)
        return DiffusionOutput(
            output=output_pil,
            stage_durations=None,
        )

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ) -> DiffusionOutput:
        # print(f"image: {image}")
        # print(f"req: {req}")
        multi_modal_data = req.prompts[0].get("multi_modal_data", None)
        image = None
        scale = None
        fidelity = None
        if multi_modal_data is not None:
            if isinstance(multi_modal_data, list):
                multi_modal_data = multi_modal_data[0]
            if "image" in multi_modal_data.keys():
                image = multi_modal_data["image"]
            if "scale" in multi_modal_data.keys():
                scale = multi_modal_data["scale"]
                # print(f"scale: {scale}")
            if "fidelity" in multi_modal_data.keys():
                fidelity = multi_modal_data["fidelity"]
                # print(f"fidelity: {fidelity}")
            if isinstance(image, list):
                image = image[0]
        
        # image = multi_modal_data[0]
        extracted_prompt, negative_prompt = self._extract_prompts(req.prompts)
        prompt = extracted_prompt or prompt
        print(f"prompt: {prompt}")

        print(f"image: {image} {type(image)}")
        print(f"scale: {scale}")
        print(f"fidelity: {fidelity}")
        if image is None:
            return DiffusionOutput(
                output=None,
                stage_durations=None,
            )

        tilesize = 64
        tile_stride = tilesize - tilesize // 4
        
        img = image.convert('RGB')
        w,h = img.size
        w_desti = round(w * scale)
        h_desti = round(h * scale)
        upsampled_img = img.resize((w_desti, h_desti), Image.BICUBIC)

        def adaptive_pad(img, tilesize, stride):
            w, h = img.size
            pad_h = (tilesize - h) if h <= tilesize else ((h - tilesize + stride - 1) // stride) * stride + tilesize - h
            pad_w = (tilesize - w) if w <= tilesize else ((w - tilesize + stride - 1) // stride) * stride + tilesize - w

            new_w = w + pad_w
            new_h = h + pad_h

            new_img = Image.new(img.mode, (new_w, new_h), color=0)
            new_img.paste(img, (0, 0))
            return new_img
        
        img = adaptive_pad(upsampled_img, tilesize=tilesize * 8, stride=tile_stride * 8)
        prompt = prompt or """High Contrast, hyper detailed photo, 2k UHD"""

        return self.infer(
            prompt=prompt,
            negative_prompt="",
            upsampled_img=upsampled_img,
            condition_image=img,
            cfg_scale=1.0,
            fidelity=fidelity,
            tiled=True,
            tile_size=tilesize,
            tile_stride=tile_stride,
            w_desti=w_desti,
            h_desti=h_desti,
        )




        # raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None

        # assert False  

        # height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        # width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        # num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        # sigmas = req.sampling_params.sigmas or sigmas
        # max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        # generator = req.sampling_params.generator or generator
        # true_cfg_scale = req.sampling_params.true_cfg_scale or true_cfg_scale
        # if req.sampling_params.guidance_scale_provided:
        #     guidance_scale = req.sampling_params.guidance_scale
        # num_images_per_prompt = (
        #     req.sampling_params.num_outputs_per_prompt
        #     if req.sampling_params.num_outputs_per_prompt > 0
        #     else num_images_per_prompt
        # )

        # ctx = self._prepare_generation_context(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     height=height,
        #     width=width,
        #     num_inference_steps=num_inference_steps,
        #     sigmas=sigmas,
        #     guidance_scale=guidance_scale,
        #     num_images_per_prompt=num_images_per_prompt,
        #     generator=generator,
        #     true_cfg_scale=true_cfg_scale,
        #     max_sequence_length=max_sequence_length,
        #     prompt_embeds=prompt_embeds,
        #     prompt_embeds_mask=prompt_embeds_mask,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        #     latents=latents,
        #     attention_kwargs=attention_kwargs,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        # )

        # latents = self.diffuse(
        #     ctx["prompt_embeds"],
        #     ctx["prompt_embeds_mask"],
        #     ctx["negative_prompt_embeds"],
        #     ctx["negative_prompt_embeds_mask"],
        #     ctx["latents"],
        #     ctx["img_shapes"],
        #     ctx["txt_seq_lens"],
        #     ctx["negative_txt_seq_lens"],
        #     ctx["timesteps"],
        #     ctx["do_true_cfg"],
        #     ctx["guidance"],
        #     true_cfg_scale,
        #     image_latents=None,
        #     cfg_normalize=True,
        #     additional_transformer_kwargs={
        #         "return_dict": False,
        #         "attention_kwargs": self.attention_kwargs,
        #     },
        # )

        # self._current_timestep = None
        # return self._decode_latents(latents, height, width, output_type)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weight_dtype = torch.bfloat16

        pretrained_qwen_path = f"{self.od_config.model}"
        trained_ckpt = f"{self.od_config.model}/ODTSR/weight.pth"

        sd_safe_tensor_path_json_format = f'''[
            [
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
                "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
            ],
            [
                "{pretrained_qwen_path}/text_encoder/model-00001-of-00004.safetensors",
                "{pretrained_qwen_path}/text_encoder/model-00002-of-00004.safetensors",
                "{pretrained_qwen_path}/text_encoder/model-00003-of-00004.safetensors",
                "{pretrained_qwen_path}/text_encoder/model-00004-of-00004.safetensors"
            ],
            "{pretrained_qwen_path}/vae/diffusion_pytorch_model.safetensors"
        ]'''

        self.model = Generator(
            torch_dtype = torch.bfloat16,
            pretrained_weights=sd_safe_tensor_path_json_format,
            tokenizer_path = f"{pretrained_qwen_path}/tokenizer",
            learning_rate=0,
            use_gradient_checkpointing=False,
            pretrained_ckpt_path_gen=trained_ckpt
        )
        self.model = self.model.to(device="cuda")
        self.model.device = next(self.model.parameters()).device
        self.model.pipe.device = self.model.device
        # loader = AutoWeightsLoader(self)
        # return loader.load_weights(weights)
        # loaded_weights = loader.load_weights(weights)
        




        # print(f"[start] load lora weight")
        # self.lora_state_dict = torch.load(f"{self.od_config.model}/ODTSR/weight.pth", map_location='cpu')
        # self.lora_state_dict_remove_prefix = {}
        # for k, v in self.lora_state_dict.items():
        #     name, module = k, v
        #     if k.startswith("pipe.dit."):
        #         name = k[len("pipe.dit."):]
        #         # self.lora_state_dict_remove_prefix[k[len("pipe.dit."):]] = v
        #     elif k.startswith("pipe."):
        #         name = k[len("pipe."):]
        #         # self.lora_state_dict_remove_prefix[k[len("pipe."):]] = v
        #     # else:
        #     #     self.lora_state_dict_remove_prefix[k] = v
        #     if "attn.to_q.lora" in name:
        #         name = name.replace("attn.to_q.lora", "attn.to_qkv.q_lora")
        #     if "attn.to_k.lora" in name:
        #         name = name.replace("attn.to_k.lora", "attn.to_qkv.k_lora")
        #     if "attn.to_v.lora" in name:
        #         name = name.replace("attn.to_v.lora", "attn.to_qkv.v_lora")
        #     if "to_out.0" in name:
        #         name = name.replace("to_out.0", "to_out")
            
        #     self.lora_state_dict_remove_prefix[name] = module

        # print(f"[end] load lora weight")
        




        # print(f"[start] linear of dit -> dual_lora linear")
        # self.add_custom_dual_lora(self.transformer, lora_rank=self.lora_rank)
        # print(f"[end] linear of dit -> dual_lora linear")




        # # total = 0
        # def replace_lora_module(parent, name_prefix=""):
        #     total = 0
        #     for name, module in parent.named_children():
        #         full_name = f"{name_prefix}{name}"
        #         if full_name + ".weight" in self.lora_state_dict_remove_prefix.keys():
        #             print(f"[replace_lora_module]: {full_name}")
        #             total = total + 1
        #             module.weight.data = self.lora_state_dict_remove_prefix[full_name + ".weight"]
        #         else:
        #             # print(f"[no replace_lora_module]: {full_name}")
        #             total = total + replace_lora_module(module, full_name + ".")
        #     return total
        # print(f"[start] replace_lora_module")
        # total = replace_lora_module(self.transformer)
        # print(f"[end] replace_lora_module")
        # print(f"total: {total}")


        return set()
