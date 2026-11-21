"""Z-Image contracts, exercised with a real tiny Diffusers transformer."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from diffusers import ZImageTransformer2DModel
from PIL import Image
from torch.utils.data import DataLoader

from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.plugins import load_plugins
from lightx2v_train.runtime.backend import init_backend
from lightx2v_train.utils.registry import build_model, build_trainer


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATFORM", "cpu")
    init_backend()
    load_plugins(["lightx2v_train_plugin.z_image"])
    torch.set_num_threads(1)
    torch.manual_seed(42)
    transformer = ZImageTransformer2DModel(
        in_channels=4,
        dim=32,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=16,
        axes_dims=[2, 2, 4],
        axes_lens=[128, 16, 16],
    )
    # Diffusers leaves these learned tokens as torch.empty until checkpoint loading.
    torch.nn.init.normal_(transformer.x_pad_token, std=0.02)
    torch.nn.init.normal_(transformer.cap_pad_token, std=0.02)
    transformer.save_pretrained(tmp_path / "model/transformer")
    vae_dir = tmp_path / "model/vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKL",
                "block_out_channels": [32, 64, 64, 64],
                "latent_channels": 4,
                "scaling_factor": 0.5,
                "shift_factor": 0.25,
            }
        )
    )
    config = {
        "plugins": ["lightx2v_train_plugin.z_image"],
        "model": {
            "name": "z_image",
            "pretrained_model_name_or_path": str(tmp_path / "model"),
            "running_dtype": "fp32",
            "transformer_param_dtype": "fp32",
            "latent_dtype": "fp32",
            "vae_dtype": "fp32",
            "max_sequence_length": 32,
        },
        "scheduler": {
            "num_train_timesteps": 1000,
            "timestep_distribution": "logitnormal",
        },
        "training": {
            "method": "flow_matching",
            "train_type": "lora",
            "max_train_iters": 2,
            "lr_warmup_iters": 0,
            "output_dir": str(tmp_path / "output"),
            "gradient_accumulation_iters": 1,
            "gradient_checkpointing": True,
            "save_every_iters": 2,
            "save_total_limit": 3,
            "lora": {"rank": 2, "alpha": 4, "target_modules": ["to_q", "to_v"]},
        },
        "data": {"train": {"target_area": 32 * 48, "prompt_dropout_rate": 0.0}},
        "inference": {"method": "none"},
        "config_path": str(tmp_path / "config.yaml"),
    }
    Path(config["config_path"]).write_text(yaml.safe_dump(config))
    return config


def load_model(config):
    model = build_model(config)
    model.load_components(
        load_transformer=True, load_vae=False, load_condition_encoder=False
    )
    return model


def cached_sample():
    return {
        "inputs": {"latents": torch.randn(1, 4, 4, 4)},
        "conditioning": {"positive": {"prompt_embed": [torch.randn(3, 16)]}},
        "meta": {},
    }


class ToyVAE:
    device = torch.device("cpu")

    def __init__(self, latent):
        self.latent = latent
        self.decoded = None

    def encode(self, image):
        self.encoded = image
        return SimpleNamespace(
            latent_dist=SimpleNamespace(
                mode=lambda: self.latent,
                sample=lambda: self.latent + 1,
            )
        )

    def decode(self, latent, return_dict=False):
        self.decoded = latent
        return (latent,)


def test_vae_normalization_and_cache_mode(config):
    model = load_model(config)
    raw = torch.full((1, 4, 4, 4), 2.25)
    model.vae = ToyVAE(raw)
    sample = {"inputs": {"target_pixel_values": torch.zeros(3, 32, 32)}}
    # Official ZImagePipeline decode: latent / scaling_factor + shift_factor.
    torch.testing.assert_close(
        model.encode_to_cache_latent(sample), torch.ones_like(raw)
    )
    torch.testing.assert_close(
        model.encode_to_latent(sample), torch.full_like(raw, 1.5)
    )
    model.image_processor = SimpleNamespace(
        postprocess=lambda image, output_type: image
    )
    decoded = model.decode_latent(torch.ones_like(raw))
    torch.testing.assert_close(decoded, raw)
    assert model.vae.encoded.shape == (1, 3, 32, 32)


def test_native_time_and_velocity_contract_has_gradients(config):
    model = load_model(config)
    latent = torch.randn(1, 4, 4, 4)
    embeds = [torch.randn(3, 16)]
    sigma = torch.tensor([0.75])
    raw = model.transformer(
        [latent[0].unsqueeze(1)],
        1 - sigma,
        embeds,
        return_dict=False,
    )[0]
    prediction = model.predict_denoiser_output(latent, sigma, {"prompt_embed": embeds})
    torch.testing.assert_close(prediction, -torch.stack([value[:, 0] for value in raw]))
    prediction.square().mean().backward()
    gradients = [p.grad for p in model.transformer.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


def test_denoiser_handles_per_sample_conditions(config):
    model = load_model(config)
    latent = torch.randn(2, 4, 4, 4)
    embeds = [torch.randn(3, 16), torch.randn(5, 16)]
    prediction = model.predict_denoiser_output(
        latent, torch.tensor([0.2, 0.8]), {"prompt_embed": embeds}
    )
    assert prediction.shape == latent.shape
    separate = torch.cat(
        [
            model.predict_denoiser_output(
                latent[i : i + 1], torch.tensor([sigma]), {"prompt_embed": [embeds[i]]}
            )
            for i, sigma in enumerate((0.2, 0.8))
        ]
    )
    torch.testing.assert_close(prediction, separate, atol=1e-5, rtol=1e-5)


def test_image_inference_cfg_matches_native_z_image(config):
    from lightx2v_train.infer.base import BaseInferencer

    model = load_model(config)
    latent = torch.randn(1, 4, 4, 4)
    sigma = torch.tensor([0.6])
    positive = {"prompt_embed": [torch.randn(3, 16)]}
    negative = {"prompt_embed": [torch.randn(5, 16)]}
    # Compare the host's inference boundary against the native pipeline's
    # guidance on raw DiT outputs, before its velocity-sign conversion.
    images = [latent[0].unsqueeze(1)]
    raw_positive = model.transformer(
        images, 1 - sigma, positive["prompt_embed"], return_dict=False
    )[0][0]
    raw_negative = model.transformer(
        images, 1 - sigma, negative["prompt_embed"], return_dict=False
    )[0][0]
    inferencer = BaseInferencer(config)
    inferencer.set_model(model)
    for scale in (0.0, 2.0):
        inferencer.guidance_scale = scale
        prediction = inferencer.cfg_guided_denoise(latent, sigma, positive, negative)
        guided_raw = raw_positive.float() + scale * (
            raw_positive.float() - raw_negative.float()
        )
        expected = -guided_raw[:, 0].unsqueeze(0)
        torch.testing.assert_close(prediction, expected)


def test_rejects_sequence_parallel_before_loading_weights(config, monkeypatch):
    from lightx2v_train_plugin.z_image import model as model_module

    model = build_model(config)
    monkeypatch.setattr(model_module, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        model,
        "_load_condition_encoder",
        lambda _: pytest.fail("Loaded weights before checking sequence parallel"),
    )
    with pytest.raises(ValueError, match="does not support sequence parallel"):
        model.load_components(
            load_transformer=True, load_vae=True, load_condition_encoder=True
        )


@pytest.mark.parametrize(
    "image_size,expected_shape", [((96, 64), (3, 32, 48)), ((64, 96), (3, 48, 32))]
)
def test_target_preprocessing_and_dimension_validation(
    config, image_size, expected_shape
):
    from lightx2v_train_plugin.z_image.data_process import ZImageDataProcessor

    processor = ZImageDataProcessor(config)
    sample = {
        "inputs": {"target_image": Image.new("RGB", image_size, color="white")},
        "meta": {},
    }
    processed = processor(sample)
    pixels = processed["inputs"]["target_pixel_values"]
    assert pixels.shape == expected_shape
    assert processed["meta"] == {
        "target_height": expected_shape[-2],
        "target_width": expected_shape[-1],
    }
    torch.testing.assert_close(pixels, torch.ones_like(pixels))
    assert "target_image" not in processed["inputs"]
    malformed = {
        "inputs": {"target_image": Image.new("RGB", (32, 32))},
        "meta": {"target_height": 32},
    }
    with pytest.raises(ValueError, match="provided together"):
        processor(malformed)
    invalid = deepcopy(config)
    invalid["data"]["processor"] = {"size_multiple": 8}
    with pytest.raises(ValueError, match="multiple of 16"):
        ZImageDataProcessor(invalid)


def test_cache_capability_preserves_prompt_list_and_geometry(config):
    model = load_model(config)
    model.vae = ToyVAE(torch.full((1, 4, 4, 4), 2.25))
    calls = []

    def encode_prompt(**kwargs):
        calls.append(kwargs)
        return [torch.ones(3, 16)], None

    model.text_pipeline = SimpleNamespace(encode_prompt=encode_prompt)
    sample = {
        "inputs": {"target_pixel_values": torch.zeros(3, 32, 32)},
        "conditioning": {"prompt": "a test image"},
        "meta": {"id": "test"},
    }
    cache = (
        model.ensure_capabilities()
        .require(FlowMatchingSFTCapability)
        .encode_training_cache(sample)
    )
    torch.testing.assert_close(cache["inputs"]["latents"], torch.ones(1, 4, 4, 4))
    assert cache["meta"]["target_height"] == 32 and cache["meta"]["target_width"] == 32
    assert cache["conditioning"]["positive"]["prompt_embed"][0].shape == (3, 16)
    assert calls[0]["prompt"] == ["a test image"]
    assert calls[0]["do_classifier_free_guidance"] is False
    assert calls[0]["max_sequence_length"] == 32


def test_lora_training_checkpoint_and_resume(config):
    sample = cached_sample()
    loader = DataLoader(
        [sample, sample], batch_size=1, collate_fn=lambda items: items[0]
    )
    model = load_model(config)
    trainer = build_trainer(config)
    trainer.set_model(model)
    trainer.set_data(loader)
    trainer.train()
    assert trainer.current_train_iteration == 2
    assert model.transformer.gradient_checkpointing
    lora_parameters = {
        name: p for name, p in model.transformer.named_parameters() if "lora_" in name
    }
    assert lora_parameters and all(p.requires_grad for p in lora_parameters.values())
    assert all(torch.isfinite(p).all() for p in lora_parameters.values())
    assert any(
        torch.count_nonzero(p)
        for name, p in lora_parameters.items()
        if "lora_B" in name
    )
    assert all(
        p.grad is None
        for name, p in model.transformer.named_parameters()
        if "lora_" not in name
    )
    checkpoint = Path(config["training"]["output_dir"]) / "checkpoint-000000002"
    assert (checkpoint / "pytorch_lora_weights.safetensors").is_file()
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert state["iteration"] == 2 and state["optimizer"]["state"]
    inference_model = load_model(config)
    inference_model.load_lora_for_infer(checkpoint)
    probe = torch.randn(1, 4, 4, 4)
    condition = {"prompt_embed": [torch.randn(3, 16)]}
    model.transformer.eval()
    inference_model.transformer.eval()
    torch.testing.assert_close(
        inference_model.predict_denoiser_output(probe, torch.tensor([0.4]), condition),
        model.predict_denoiser_output(probe, torch.tensor([0.4]), condition),
    )
    resumed = deepcopy(config)
    resumed["resume"] = {"auto_resume": True}
    resumed["training"]["max_train_iters"] = 3
    resumed["training"]["save_every_iters"] = 1
    resumed_trainer = build_trainer(resumed)
    resumed_trainer.set_model(load_model(resumed))
    resumed_trainer.set_data(loader)
    resumed_trainer.train()
    assert resumed_trainer.current_train_iteration == 3
    state = torch.load(
        checkpoint.parent / "checkpoint-000000003/training_state.pt", weights_only=True
    )
    assert all(value["step"] == 3 for value in state["optimizer"]["state"].values())
