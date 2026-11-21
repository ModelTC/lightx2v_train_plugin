import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from peft import LoraConfig, inject_adapter_in_model
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "LightX2V-Train-Plugin/lightx2v_train")]

from lightx2v_train.plugins import load_plugins
from lightx2v_train.runtime.backend import init_backend
from lightx2v_train.utils.registry import build_model, build_trainer


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATFORM", "cpu")
    init_backend()
    load_plugins(["lightx2v_train_plugin.wan22"])
    from lightx2v_train.model_zoo.native.wan.modules.model import WanModel

    torch.manual_seed(42)
    torch.set_num_threads(1)
    transformer = WanModel(dim=32, ffn_dim=64, freq_dim=16, num_heads=4, num_layers=1, text_len=4)
    torch.nn.init.normal_(transformer.head.head.weight, std=0.02)
    for expert in ("high", "low"):
        transformer.save_pretrained(tmp_path / "model" / f"{expert}_noise_model")
    config = {
        "plugins": ["lightx2v_train_plugin.wan22"],
        "model": {"name": "wan22_t2v_a14b", "expert": "high", "pretrained_model_name_or_path": str(tmp_path / "model"), "running_dtype": "bf16"},
        "scheduler": {"num_train_timesteps": 1000, "shift": 5},
        "training": {
            "method": "wan22_sft", "train_type": "lora", "max_train_iters": 2,
            "lr_warmup_iters": 5, "output_dir": str(tmp_path / "output"),
            "gradient_accumulation_iters": 1, "gradient_checkpointing": True,
            "gradient_checkpointing_offload": True, "save_every_iters": 2, "save_total_limit": 5,
            "lora": {"rank": 2, "alpha": 2, "target_modules": ["q", "v"]},
        },
        "inference": {"method": "none", "num_frames": 5},
        "runtime": {"seed": 42},
        "config_path": str(tmp_path / "config.yaml"),
    }
    Path(config["config_path"]).write_text(yaml.safe_dump(config))
    return config


def load_model(config):
    model = build_model(config)
    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
    return model


def test_selected_expert_and_peft_initialization(config):
    model = load_model(config)
    assert model.transformer_path.endswith("high_noise_model")
    assert model.vae is None and model.text_encoder is None
    reference = deepcopy(model.transformer)
    torch.manual_seed(123)
    reference = inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=["q", "v"]), reference)
    torch.manual_seed(123)
    model.add_lora(2, 2, ["q", "v"])
    expected = dict(reference.named_parameters())
    for name, parameter in model.transformer.named_parameters():
        if "lora_" in name:
            assert parameter.dtype == torch.bfloat16
            torch.testing.assert_close(parameter, expected[name].to(torch.bfloat16), rtol=0, atol=0)
    low = deepcopy(config)
    low["model"]["expert"] = "low"
    assert load_model(low).transformer_path.endswith("low_noise_model")


def test_native_wan_training_and_resume(config):
    torch.manual_seed(123)
    sample = {
        "inputs": {"latents": torch.randn(1, 16, 1, 4, 4, dtype=torch.bfloat16)},
        "conditioning": {"positive": {"prompt_embed": torch.randn(1, 4, 4096, dtype=torch.bfloat16)}},
        "meta": {},
    }
    loader = DataLoader([sample, sample], batch_size=1, collate_fn=lambda samples: samples[0])
    model = load_model(config)
    trainer = build_trainer(config)
    trainer.set_model(model)
    trainer.set_data(loader)
    trainer.train()
    assert trainer.current_train_iteration == 2
    assert any(torch.count_nonzero(p) for n, p in model.transformer.named_parameters() if "lora_B" in n)
    checkpoint = Path(config["training"]["output_dir"]) / "checkpoint-000000002"
    assert (checkpoint / "pytorch_lora_weights.safetensors").is_file()
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert state["iteration"] == 2 and state["optimizer"]["state"]

    resumed = deepcopy(config)
    resumed["resume"] = {"auto_resume": True}
    resumed["training"]["max_train_iters"] = 3
    resumed["training"]["save_every_iters"] = 1
    resumed_trainer = build_trainer(resumed)
    resumed_trainer.set_model(load_model(resumed))
    resumed_trainer.set_data(loader)
    resumed_trainer.train()
    assert resumed_trainer.current_train_iteration == 3
    state = torch.load(checkpoint.parent / "checkpoint-000000003/training_state.pt", weights_only=True)
    assert all(item["step"] == 3 for item in state["optimizer"]["state"].values())

    from lightx2v_train_plugin.wan22.trainer import Wan22SFTTrainer

    wrong_expert = Wan22SFTTrainer.__new__(Wan22SFTTrainer)
    wrong_expert.model_config = {"expert": "low"}
    with pytest.raises(RuntimeError, match="expert checkpoint"):
        wrong_expert._validate_checkpoint_metadata({"world_size": 1, "iteration": 2}, "test", checkpoint)


def test_learning_rates_and_cpu_inference_noise(config):
    trainer = build_trainer(config)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = trainer._build_lr_scheduler(optimizer)
    used = []
    for _ in range(6):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert used == pytest.approx([2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 1e-4])

    model = load_model(config)
    latent = model.prepare_infer_latents(32, 32, torch.Generator().manual_seed(7))
    expected = torch.randn((1, 16, 2, 4, 4), generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    torch.testing.assert_close(latent, expected, rtol=0, atol=0)


def test_saved_tensor_offload_preserves_native_gradients(config):
    model = load_model(config)
    model.add_lora(2, 2, ["q", "v"])
    model.set_lora_trainable()
    model.enable_gradient_checkpointing()
    latent = torch.randn(1, 16, 1, 4, 4, dtype=torch.bfloat16)
    condition = {"prompt_embed": torch.randn(1, 4, 4096, dtype=torch.bfloat16)}
    model.config["training"]["gradient_checkpointing_offload"] = False
    model.predict_denoiser_output(latent, torch.tensor([0.9]), condition).square().mean().backward()
    expected = {name: p.grad.clone() for name, p in model.transformer.named_parameters() if p.requires_grad}
    model.transformer.zero_grad(set_to_none=True)
    model.config["training"]["gradient_checkpointing_offload"] = True
    model.predict_denoiser_output(latent, torch.tensor([0.9]), condition).square().mean().backward()
    for name, p in model.transformer.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, expected[name], rtol=0, atol=0)
        else:
            assert p.grad is None


def test_inference_uses_configured_negative_prompt_and_boundary(config):
    from lightx2v_train_plugin.wan22.infer import Wan22T2VInferencer

    config["inference"].update(negative_prompt="reference negative prompt", boundary_sigma=0.875)
    inferencer = Wan22T2VInferencer(config)
    assert inferencer.negative_prompt == "reference negative prompt"
    inferencer.model, inferencer.low_model = object(), object()
    assert inferencer._denoise_model_for_step(0, 50, torch.tensor(0.875)) is inferencer.model
    assert inferencer._denoise_model_for_step(1, 50, torch.tensor(0.874)) is inferencer.low_model
    sigmas = inferencer._inference_sigmas(50)
    assert len(sigmas) == 50 and sigmas[0] == 1 and all(a > b for a, b in zip(sigmas, sigmas[1:]))
    config["inference"]["sigma_shift"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        inferencer._inference_sigmas(50)
