# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.cuda.amp import autocast

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel

# Try to use torch.compile for speed optimization
try:
    torch._dynamo.config.suppress_errors = True
    compile_available = hasattr(torch, 'compile')
except:
    compile_available = False

# Import FlexAttention support
try:
    from ...flex_attention import create_attention_processor, check_flex_attention
    FLEX_SUPPORT = True
except:
    FLEX_SUPPORT = False
    create_attention_processor = None

# Import adaptive memory optimizer
try:
    from ...adaptive_memory_optimizer import create_adaptive_optimizer, integrate_adaptive_optimizer
    ADAPTIVE_MEMORY = True
except:
    ADAPTIVE_MEMORY = False
    create_adaptive_optimizer = None
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")
        
        # Compile critical functions for speed on supported GPUs
        # Can be disabled via DISABLE_TORCH_COMPILE=1 for PyTorch 2.7.1 compatibility
        if compile_available and torch.cuda.is_available() and os.environ.get('DISABLE_TORCH_COMPILE', '0') != '1':
            try:
                # Compile the UNet for faster inference
                self.unet = torch.compile(self.unet, mode="reduce-overhead")
                print("Successfully compiled UNet for faster inference")
            except Exception as e:
                print(f"Could not compile UNet: {e}")
        
        # Store default attention mode
        self.attention_mode = None
        self.attention_processor = None
        
        # Initialize adaptive memory optimizer
        if ADAPTIVE_MEMORY:
            try:
                self.memory_optimizer = create_adaptive_optimizer()
                print("✓ Adaptive memory optimization enabled")
            except Exception as e:
                print(f"Could not enable adaptive memory optimization: {e}")
                self.memory_optimizer = None
        else:
            self.memory_optimizer = None

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        
        # Decode with adaptive batch size
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            batch_size = self.memory_optimizer.get_optimal_batch_size('vae_decode')
        else:
            batch_size = 4  # Default fallback
        decoded_chunks = []
        
        for i in range(0, latents.shape[0], batch_size):
            chunk = latents[i:i+batch_size]
            decoded_chunk = self.vae.decode(chunk).sample
            decoded_chunks.append(decoded_chunk)
            
            # Clear cache after each chunk to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        decoded_latents = torch.cat(decoded_chunks, dim=0)
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    @torch.no_grad()
    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        
        # Batch processing with adaptive sizing
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            batch_size = self.memory_optimizer.get_optimal_batch_size('face_processing')
        else:
            batch_size = 8  # Default fallback
        num_frames = len(video_frames)
        
        # Process in batches with progress bar
        with tqdm.tqdm(total=num_frames) as pbar:
            for i in range(0, num_frames, batch_size):
                batch_end = min(i + batch_size, num_frames)
                batch_frames = video_frames[i:batch_end]
                
                # Process batch
                for frame in batch_frames:
                    try:
                        face, box, affine_matrix = self.image_processor.affine_transform(frame)
                        faces.append(face)
                        boxes.append(box)
                        affine_matrices.append(affine_matrix)
                    except Exception as e:
                        print(f"Warning: Face detection failed for frame: {e}")
                        # Use a blank face if detection fails
                        blank_face = torch.zeros(3, self.image_processor.resolution, self.image_processor.resolution)
                        faces.append(blank_face)
                        boxes.append([0, 0, self.image_processor.resolution, self.image_processor.resolution])
                        affine_matrices.append(np.eye(2, 3))
                
                # Clear GPU cache periodically to prevent memory buildup
                if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.update(batch_end - i)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    @torch.no_grad()
    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        
        # Batch processing with adaptive sizing
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            batch_size = self.memory_optimizer.get_optimal_batch_size('face_restoration')
        else:
            batch_size = 8  # Default fallback
        num_faces = len(faces)
        
        # Pre-allocate output array for better memory efficiency
        out_frames = np.zeros_like(video_frames[:num_faces])
        
        with tqdm.tqdm(total=num_faces) as pbar:
            for i in range(0, num_faces, batch_size):
                batch_end = min(i + batch_size, num_faces)
                
                # Process batch of faces
                batch_faces = []
                for j in range(i, batch_end):
                    x1, y1, x2, y2 = boxes[j]
                    height = int(y2 - y1)
                    width = int(x2 - x1)
                    
                    # Resize face to match the detected box size
                    resized_face = torchvision.transforms.functional.resize(
                        faces[j], size=(height, width), 
                        interpolation=transforms.InterpolationMode.BICUBIC, 
                        antialias=True
                    )
                    batch_faces.append((j, resized_face))
                
                # Restore faces in batch
                for idx, face in batch_faces:
                    out_frames[idx] = self.image_processor.restorer.restore_img(
                        video_frames[idx], face, affine_matrices[idx]
                    )
                
                # Clear GPU cache periodically to prevent memory buildup
                if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.update(batch_end - i)
        
        return out_frames

    @torch.no_grad()
    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            # Process faces once for the base video
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            
            # Pre-allocate arrays for better performance
            total_frames_needed = len(whisper_chunks)
            loop_video_frames = np.zeros((total_frames_needed,) + video_frames.shape[1:], dtype=video_frames.dtype)
            loop_faces = torch.zeros((total_frames_needed,) + faces.shape[1:], dtype=faces.dtype, device=faces.device)
            loop_boxes = []
            loop_affine_matrices = []
            
            # Fill arrays efficiently
            frame_idx = 0
            for i in range(num_loops):
                if frame_idx >= total_frames_needed:
                    break
                    
                if i % 2 == 0:
                    # Forward direction
                    end_idx = min(frame_idx + len(video_frames), total_frames_needed)
                    copy_len = end_idx - frame_idx
                    loop_video_frames[frame_idx:end_idx] = video_frames[:copy_len]
                    loop_faces[frame_idx:end_idx] = faces[:copy_len]
                    loop_boxes.extend(boxes[:copy_len])
                    loop_affine_matrices.extend(affine_matrices[:copy_len])
                else:
                    # Reverse direction - reuse already computed faces
                    end_idx = min(frame_idx + len(video_frames), total_frames_needed)
                    copy_len = end_idx - frame_idx
                    loop_video_frames[frame_idx:end_idx] = video_frames[:copy_len][::-1]
                    loop_faces[frame_idx:end_idx] = faces[:copy_len].flip(0)
                    loop_boxes.extend(boxes[:copy_len][::-1])
                    loop_affine_matrices.extend(affine_matrices[:copy_len][::-1])
                
                frame_idx = end_idx
            
            video_frames = loop_video_frames
            faces = loop_faces
            boxes = loop_boxes[:total_frames_needed]
            affine_matrices = loop_affine_matrices[:total_frames_needed]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    def set_attention_mode(self, mode: str):
        """Set the attention mechanism to use"""
        self.attention_mode = mode
        
        if mode == "flex":
            if FLEX_SUPPORT:
                self.attention_processor = create_attention_processor(mode="flex")
                if self.attention_processor:
                    print("Using FlexAttention with lip-sync optimizations")
                else:
                    print("FlexAttention not available, using standard attention")
            else:
                print("FlexAttention support not installed, using standard attention")
        
        elif mode == "flash" or mode == "xformers":
            if hasattr(self.unet, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.unet.enable_xformers_memory_efficient_attention()
                    print("Enabled Flash/xformers memory efficient attention")
                except Exception as e:
                    print(f"Could not enable xformers: {e}")
        
        elif mode == "standard":
            if hasattr(self.unet, 'disable_xformers_memory_efficient_attention'):
                try:
                    self.unet.disable_xformers_memory_efficient_attention()
                    print("Using standard attention")
                except:
                    pass
    
    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)

        video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)

        # Instead of accumulating frames in memory, we'll write to disk
        import tempfile
        import cv2
        
        # Create temporary directory for processed frames
        # Use system temp if no temp_dir provided
        base_temp = getattr(self, 'temp_dir', None) or tempfile.gettempdir()
        temp_output_dir = os.path.join(base_temp, f"latentsync_frames_{os.getpid()}")
        os.makedirs(temp_output_dir, exist_ok=True)
        frame_index = 0
        
        # Don't accumulate frames in memory anymore
        # synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels
        
        # Initialize CUDA graph for UNet if possible
        # Disable CUDA graphs due to PyTorch 2.7.1 compatibility issues with cudaMallocAsync
        use_cuda_graph = False  # Temporarily disabled to fix checkPoolLiveAllocations error
        cuda_graph = None
        graph_initialized = False
        
        # Set attention mode if provided
        attention_mode = kwargs.get('attention_mode', 'flex')
        if attention_mode != self.attention_mode:
            self.set_attention_mode(attention_mode)

        # Enable memory format optimization for tensor cores
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Apply adaptive optimization to pipeline if available
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            video_specs = {
                'length': len(whisper_chunks),
                'resolution': (height, width),
                'total_pixels': len(whisper_chunks) * height * width
            }
            self.memory_optimizer.optimize_pipeline(self, video_specs)

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            # Enhanced memory clearing to prevent end-stage lag
            progress = i / num_inferences
            
            # Progressive memory clearing - more aggressive as we progress
            if progress > 0.8:  # Last 20% of iterations
                # Very aggressive clearing every iteration
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect(2)  # Full collection
            elif progress > 0.6:  # 60-80% through
                # Moderate clearing every iteration
                if i > 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            elif i > 0 and i % 2 == 0:  # First 60%
                # Normal clearing every 2 iterations
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            
            # Process masks in smaller batches to prevent OOM
            if len(inference_faces) > 8 and torch.cuda.is_available():
                # Split into smaller batches for memory safety
                ref_pixel_values_list = []
                masked_pixel_values_list = []
                masks_list = []
                
                for j in range(0, len(inference_faces), 8):
                    batch_faces = inference_faces[j:j+8]
                    ref_pv, masked_pv, m = self.image_processor.prepare_masks_and_masked_images(
                        batch_faces, affine_transform=False
                    )
                    ref_pixel_values_list.append(ref_pv)
                    masked_pixel_values_list.append(masked_pv)
                    masks_list.append(m)
                
                ref_pixel_values = torch.cat(ref_pixel_values_list, dim=0)
                masked_pixel_values = torch.cat(masked_pixel_values_list, dim=0)
                masks = torch.cat(masks_list, dim=0)
                
                # Clean up intermediate tensors
                del ref_pixel_values_list, masked_pixel_values_list, masks_list
                torch.cuda.empty_cache()
            else:
                ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    inference_faces, affine_transform=False
                )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat(
                        [unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                    )

                    # predict the noise residual with mixed precision for speed
                    with autocast(enabled=torch.cuda.is_available()):
                        # Apply custom attention processor if using FlexAttention
                        if self.attention_processor and self.attention_mode == "flex":
                            # Temporarily set the processor for this forward pass
                            old_processor = getattr(self.unet, '_attention_processor', None)
                            self.unet._attention_processor = self.attention_processor
                            
                        noise_pred = self.unet(
                            unet_input, t, encoder_hidden_states=audio_embeds
                        ).sample
                        
                        # Restore old processor if changed
                        if self.attention_processor and self.attention_mode == "flex":
                            if old_processor is not None:
                                self.unet._attention_processor = old_processor
                            elif hasattr(self.unet, '_attention_processor'):
                                del self.unet._attention_processor

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            # Write frames to disk instead of accumulating in memory
            # First restore the frames for this chunk
            chunk_start = i * num_frames
            chunk_end = min((i + 1) * num_frames, len(video_frames))
            chunk_video_frames = video_frames[chunk_start:chunk_end]
            chunk_boxes = boxes[chunk_start:chunk_end]
            chunk_affine_matrices = affine_matrices[chunk_start:chunk_end]
            
            # Restore video for this chunk
            restored_chunk = self.restore_video(
                decoded_latents, 
                chunk_video_frames, 
                chunk_boxes, 
                chunk_affine_matrices
            )
            
            # Write frames to disk
            for j, frame in enumerate(restored_chunk):
                frame_path = os.path.join(temp_output_dir, f"frame_{frame_index:06d}.png")
                
                # Debug: log frame info for first frame of first chunk
                if frame_index == 0:
                    print(f"Debug - Frame type: {type(frame)}, dtype: {frame.dtype if hasattr(frame, 'dtype') else 'N/A'}")
                    print(f"Debug - Frame shape: {frame.shape if hasattr(frame, 'shape') else 'N/A'}")
                    if hasattr(frame, 'max') and hasattr(frame, 'min'):
                        print(f"Debug - Frame range: [{frame.min():.3f}, {frame.max():.3f}]")
                
                # Convert to numpy if it's a tensor, otherwise it's already numpy
                if torch.is_tensor(frame):
                    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                else:
                    # Already numpy array - check if it needs scaling
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        # Float array, scale it
                        frame_np = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    else:
                        # Already uint8
                        frame_np = frame
                
                # The restored frames are in RGB format, convert to BGR for cv2
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frame_path, frame_bgr)
                frame_index += 1
            
            # Clear ALL memory including the decoded frames
            if 'latents' in locals():
                del latents
            if 'mask_latents' in locals():
                del mask_latents
            if 'masked_image_latents' in locals():
                del masked_image_latents
            if 'ref_latents' in locals():
                del ref_latents
            if 'decoded_latents' in locals():
                del decoded_latents
            if 'noise_pred' in locals():
                del noise_pred
            if 'restored_chunk' in locals():
                del restored_chunk
            if 'unet_input' in locals():
                del unet_input
            
            # Force garbage collection every few iterations
            if (i + 1) % 3 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

        # All frames have been written to disk, now create video from frames
        # Count total frames processed
        total_frames = frame_index
        
        audio_samples_remain_length = int(total_frames / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        # Use the same base temp as frames for consistency
        temp_dir = os.path.join(base_temp, f"latentsync_final_{os.getpid()}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # Create video from frames on disk using ffmpeg (no memory usage)
        video_path = os.path.join(temp_dir, "video.mp4")
        frame_pattern = os.path.join(temp_output_dir, "frame_%06d.png")
        
        try:
            # Use hardware encoding if available for faster video creation
            # First try NVIDIA hardware encoder
            encoders = [
                (["ffmpeg", "-y", "-r", str(video_fps), "-i", frame_pattern,
                  "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                  "-pix_fmt", "yuv420p", video_path], "h264_nvenc"),
                (["ffmpeg", "-y", "-r", str(video_fps), "-i", frame_pattern,
                  "-c:v", "libx264", "-preset", "faster", "-crf", "18",
                  "-pix_fmt", "yuv420p", video_path], "libx264")
            ]
            
            for cmd, encoder_name in encoders:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"Video encoded successfully with {encoder_name}")
                    break
                except subprocess.CalledProcessError:
                    if encoder_name == "libx264":
                        raise  # Re-raise if software encoder fails
                    continue  # Try next encoder

            sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

            # Fix: Use list of arguments instead of shell=True to prevent injection
            command = [
                "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                "-i", os.path.join(temp_dir, "video.mp4"),
                "-i", os.path.join(temp_dir, "audio.wav"),
                "-c:v", "libx264", "-crf", "18",
                "-c:a", "aac", "-q:v", "0", "-q:a", "0",
                video_out_path
            ]
            subprocess.run(command, check=True)
        finally:
            # Always clean up temporary directories
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
