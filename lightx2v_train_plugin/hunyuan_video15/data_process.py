"""Prepare still-image targets and validate HunyuanVideo 1.5 geometry."""

import math

import numpy as np
import torch
from PIL import Image
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


def validate_video_shape(frames, height, width):
    if frames < 1 or (frames - 1) % 4:
        raise ValueError(f"HunyuanVideo 1.5 requires 4n+1 video frames, got {frames}.")
    if height < 16 or width < 16 or height % 16 or width % 16:
        raise ValueError(f"HunyuanVideo 1.5 height/width must be positive multiples of 16, got {height}x{width}.")


@SAMPLE_PROCESSOR_REGISTER("hunyuan_video15_t2v")
class HunyuanVideo15DataProcessor:
    image_mode = "RGB"

    def __init__(self, config):
        self.unconditional_prompt = config["model"].get("unconditional_prompt", "")
        processor = config.get("data", {}).get("processor", {})
        self.image_height = int(processor.get("height", 512))
        self.image_width = int(processor.get("width", 512))
        for split in ("train", "val"):
            data = config.get("data", {}).get(split, {})
            if data.get("name") == "video_dataset":
                validate_video_shape(data.get("num_frames", 81), data.get("height", 480), data.get("width", 832))

    def __call__(self, sample):
        inputs = sample.get("inputs", {})
        image = inputs.get("target_image")
        if image is not None:
            if not isinstance(image, Image.Image):
                raise TypeError("HunyuanVideo 1.5 inputs.target_image must be a PIL image.")
            if inputs.get("video") is not None or inputs.get("source_images"):
                raise ValueError("HunyuanVideo 1.5 image SFT expects only a target image, without video or source_images.")
            validate_video_shape(1, self.image_height, self.image_width)
            image = image.convert(self.image_mode)
            scale = max(self.image_width / image.width, self.image_height / image.height)
            width = max(self.image_width, math.ceil(image.width * scale))
            height = max(self.image_height, math.ceil(image.height * scale))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            left = (width - self.image_width) // 2
            top = (height - self.image_height) // 2
            image = image.crop((left, top, left + self.image_width, top + self.image_height))
            pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            inputs["video"] = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1).contiguous()
            inputs.pop("target_image")
            sample.setdefault("meta", {}).update(
                target_height=self.image_height,
                target_width=self.image_width,
                num_frames=1,
            )
        video = inputs.get("video")
        if video is not None:
            if video.ndim not in (4, 5) or video.shape[-4] != 3:
                raise ValueError("HunyuanVideo 1.5 video must have shape [3,F,H,W] or [B,3,F,H,W].")
            validate_video_shape(*video.shape[-3:])
        return sample
