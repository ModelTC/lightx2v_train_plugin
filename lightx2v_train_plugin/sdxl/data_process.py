"""Image resizing and crop coordinates for SDXL."""

import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from lightx2v_train.utils.image_ops import align_dimension, calculate_area_dimensions
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


def _scalar(value, name):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one value.")
        value = value.item()
    elif isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain one value.")
        return _scalar(value[0], name)
    return int(value)


@SAMPLE_PROCESSOR_REGISTER("sdxl")
class SDXLDataProcessor:
    image_mode = "RGB"

    def __init__(self, config):
        self.config = config
        data = config.get("data", {})
        processor = data.get("processor", {})
        self.size_multiple = int(processor.get("size_multiple", 64))
        if self.size_multiple <= 0 or self.size_multiple % 64:
            raise ValueError("SDXL size_multiple must be a positive multiple of 64.")
        area = processor.get("target_area")
        if area is None:
            area = data.get("train", {}).get("target_area", 1024 * 1024)
        self.target_area = int(area)
        if self.target_area <= 0:
            raise ValueError("SDXL target_area must be positive.")
        self.unconditional_prompt = config["model"].get("unconditional_prompt", "")
        self.image_processor = VaeImageProcessor(vae_scale_factor=8, do_resize=False)

    @staticmethod
    def _explicit_target_size(sample):
        meta = sample.get("meta", {})
        height, width = meta.get("target_height"), meta.get("target_width")
        if (height is None) != (width is None):
            raise ValueError("meta.target_height and meta.target_width must be provided together.")
        if height is None:
            return None, None
        return _scalar(height, "target_height"), _scalar(width, "target_width")

    def infer_target_size(self, sample, default_height, default_width):
        height, width = self._explicit_target_size(sample)
        height = int(default_height) if height is None else height
        width = int(default_width) if width is None else width
        return align_dimension(height, self.size_multiple), align_dimension(width, self.size_multiple)

    @staticmethod
    def _record_geometry(meta, original_height, original_width, top, left, height, width):
        meta.update(
            original_height=int(original_height),
            original_width=int(original_width),
            crop_top=int(top),
            crop_left=int(left),
            target_height=int(height),
            target_width=int(width),
        )
        meta["time_ids"] = torch.tensor([original_height, original_width, top, left, height, width], dtype=torch.int64)

    def __call__(self, sample):
        inputs = sample["inputs"]
        meta = sample.setdefault("meta", {})
        if inputs.get("source_images"):
            raise ValueError("SDXL Base SFT expects target images and text; source-image editing is not supported.")
        image = inputs.pop("target_image", None)
        if image is None:
            inference = self.config.get("inference", {})
            height, width = self.infer_target_size(sample, inference.get("default_height", 1024), inference.get("default_width", 1024))
            self._record_geometry(meta, height, width, 0, 0, height, width)
            return sample

        image = image.convert(self.image_mode)
        original_width, original_height = image.size
        height, width = self._explicit_target_size(sample)
        if height is None:
            width, height = calculate_area_dimensions(self.target_area, original_width / original_height, self.size_multiple)
        else:
            height, width = self.infer_target_size(sample, height, width)
        scale = max(width / original_width, height / original_height)
        resized_width = max(width, round(original_width * scale))
        resized_height = max(height, round(original_height * scale))
        image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        inputs["target_pixel_values"] = self.image_processor.preprocess(image)[0]
        self._record_geometry(meta, original_height, original_width, top, left, height, width)
        return sample
