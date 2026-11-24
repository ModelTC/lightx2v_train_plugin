"""Image preprocessing for Z-Image training."""

from diffusers.image_processor import VaeImageProcessor
from lightx2v_train.utils.image_ops import (
    align_dimension,
    calculate_area_dimensions,
    resize_and_center_crop,
)
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER


@SAMPLE_PROCESSOR_REGISTER("z_image")
class ZImageDataProcessor:
    """Resize RGB targets to the 16-pixel geometry required by Z-Image."""

    image_mode = "RGB"

    def __init__(self, config):
        processor_config = config.get("data", {}).get("processor", {})
        size_multiple = int(processor_config.get("size_multiple", 16))
        if size_multiple <= 0 or size_multiple % 16:
            raise ValueError("Z-Image size_multiple must be a positive multiple of 16")
        self.image_processor = VaeImageProcessor(vae_scale_factor=size_multiple)
        self.size_multiple = size_multiple
        self.target_area = self._target_area(config)
        self.unconditional_prompt = config["model"].get("unconditional_prompt", "")

    @staticmethod
    def _target_area(config):
        data = config.get("data", {})
        for section in ("processor", "train", "val"):
            value = data.get(section, {}).get("target_area")
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError("Z-Image target_area must be positive")
                return value
        return 1024 * 1024

    @staticmethod
    def _explicit_target_size(sample):
        meta = sample.get("meta", {})
        height = meta.get("target_height")
        width = meta.get("target_width")
        if height is None and width is None:
            return None, None
        if height is None or width is None:
            raise ValueError("meta.target_height and meta.target_width must be provided together")
        return int(height), int(width)

    def __call__(self, sample):
        image = sample["inputs"].pop("target_image", None)
        if image is None:
            return sample
        height, width = self._explicit_target_size(sample)
        if height is None:
            width, height = calculate_area_dimensions(
                self.target_area,
                image.width / image.height,
                multiple=self.size_multiple,
            )
        else:
            height = align_dimension(height, self.size_multiple)
            width = align_dimension(width, self.size_multiple)
        image = resize_and_center_crop(image.convert(self.image_mode), width, height)
        pixels = self.image_processor.preprocess(image)[0]
        sample["inputs"]["target_pixel_values"] = pixels
        sample["meta"]["target_height"] = int(pixels.shape[-2])
        sample["meta"]["target_width"] = int(pixels.shape[-1])
        return sample

    def infer_target_size(self, sample, default_height, default_width):
        height, width = self._explicit_target_size(sample)
        height = int(default_height) if height is None else height
        width = int(default_width) if width is None else width
        return (
            align_dimension(height, self.size_multiple),
            align_dimension(width, self.size_multiple),
        )
