"""CPU comparisons against the local DiffSynth source; no models are loaded."""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def capability_class():
    core = Path(os.environ.get("LIGHTX2V_TRAIN_ROOT", ROOT.parent / "LightX2V-Train-Plugin")) / "lightx2v_train"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(core))
        module = load_source("wan22_sft_under_test", ROOT / "lightx2v_train_plugin/wan22/sft.py")
    return module.Wan22SFTCapability


@pytest.fixture(scope="module")
def reference():
    root = Path(os.environ.get("DIFFSYNTH_REFERENCE_ROOT", ROOT.parent / "DiffSynth-Studio"))
    scheduler_path = root / "diffsynth/diffusion/flow_match.py"
    loss_path = root / "diffsynth/diffusion/loss.py"
    if not scheduler_path.is_file() or not loss_path.is_file():
        pytest.skip("Set DIFFSYNTH_REFERENCE_ROOT to compare against the DiffSynth checkout.")
    scheduler_module = load_source("reference_flow_match", scheduler_path)
    # Compile the unchanged loss function without importing the pipeline package.
    tree = ast.parse(loss_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "FlowMatchSFTLoss")
    namespace = {"torch": torch, "BasePipeline": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(loss_path), "exec"), namespace)
    scheduler = scheduler_module.FlowMatchScheduler("Wan")
    scheduler.set_timesteps(1000, training=True)
    return scheduler, namespace["FlowMatchSFTLoss"]


class TinyModel:
    def __init__(self, expert="high", scheduler=None):
        self.config = {"model": {"expert": expert}, "scheduler": scheduler or {"num_train_timesteps": 1000, "shift": 5.0}}
        self.device = torch.device("cpu")
        self.latent_dtype = torch.bfloat16
        self.running_dtype = torch.bfloat16
        self.scale = torch.nn.Parameter(torch.tensor(0.35))
        self.observed = None

    def forward_at_timestep(self, latent, timestep, prompt_embed):
        self.observed = (latent.detach().clone(), timestep.detach().clone())
        time = timestep.float().reshape(-1, 1, 1, 1, 1) / 1000
        prediction = latent.float() * self.scale + time + prompt_embed.float().mean() / 100
        return prediction.to(torch.bfloat16)

    def predict_denoiser_output(self, latent, sigma, condition):
        return self.forward_at_timestep(latent, sigma.float() * 1000, condition["prompt_embed"]).float()


def fixed_draws(index, noise, calls, bounds):
    def randint(low, high, size):
        assert (low, high) == bounds
        assert size == (1,)
        calls.append("index")
        return torch.tensor([index])

    def randn_like(latent):
        assert latent.dtype == torch.bfloat16
        assert latent.shape == noise.shape
        calls.append("noise")
        return noise.clone()

    return patch("torch.randint", side_effect=randint), patch("torch.randn_like", side_effect=randn_like)


@pytest.mark.parametrize("expert,bounds", [("high", (0, 417)), ("low", (417, 1000))])
def test_timestep_table_and_global_weights(capability_class, reference, expert, bounds):
    scheduler, _ = reference
    model = TinyModel(expert)
    capability = capability_class(model)
    assert (capability.timestep_start, capability.timestep_end) == bounds
    torch.testing.assert_close(capability.sigmas, scheduler.sigmas, rtol=0, atol=0)
    torch.testing.assert_close(capability.timesteps, scheduler.timesteps, rtol=0, atol=0)
    torch.testing.assert_close(capability.training_weights, scheduler.linear_timesteps_weights, rtol=0, atol=0)
    assert capability.training_weights[0] == 0
    assert capability.training_weights.mean().item() == pytest.approx(1.0)
    assert capability.training_weights[:417].mean().item() == pytest.approx(0.342366, rel=1e-5)
    assert capability.training_weights[417:].mean().item() == pytest.approx(1.470383, rel=1e-5)


@pytest.mark.parametrize(
    "expert,index",
    [("high", 0), ("high", 1), ("high", 416), ("low", 417), ("low", 418), ("low", 829), ("low", 830), ("low", 832), ("low", 833), ("low", 999)],
)
def test_weighted_loss_and_gradient_match_reference(capability_class, reference, expert, index):
    from lightx2v_train.model_capabilities import SFTStepContext

    scheduler, reference_loss = reference
    latent = torch.linspace(-2, 2, 48).reshape(1, 2, 2, 3, 4).to(torch.bfloat16)
    noise = torch.linspace(1.5, -0.5, 48).reshape_as(latent).to(torch.bfloat16)
    prompt = torch.tensor([[[0.25, -0.5]]], dtype=torch.bfloat16)
    batch = {"inputs": {"latents": latent}, "conditioning": {"positive": {"prompt_embed": prompt}}, "meta": {}}
    model = TinyModel(expert)
    reference_model = TinyModel(expert)
    capability = capability_class(model)
    bounds = (capability.timestep_start, capability.timestep_end)
    context = SFTStepContext(noise_scheduler=None, running_dtype=torch.bfloat16, broadcast=lambda value: value)

    calls = []
    randint, randn = fixed_draws(index, noise, calls, bounds)
    with randint, randn:
        result = capability.compute_loss(batch, context)
    assert calls == ["index", "noise"]

    pipe = SimpleNamespace(
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        in_iteration_models=[],
        model_fn=lambda latents, timestep, context, **_: reference_model.forward_at_timestep(latents, timestep, context),
    )
    calls = []
    randint, randn = fixed_draws(index, noise, calls, bounds)
    with randint, randn:
        expected_loss = reference_loss(
            pipe,
            input_latents=latent,
            context=prompt,
            min_timestep_boundary=bounds[0] / 1000,
            max_timestep_boundary=bounds[1] / 1000,
        )
    assert calls == ["index", "noise"]
    expected_timestep = scheduler.timesteps[index].to(torch.bfloat16)
    expected_sigma = scheduler.sigmas[torch.argmin((scheduler.timesteps - expected_timestep).abs())]
    torch.testing.assert_close(result.metrics["sigma"], expected_sigma, rtol=0, atol=0)
    torch.testing.assert_close(result.metrics["loss_weight"], scheduler.training_weight(expected_timestep), rtol=0, atol=0)
    torch.testing.assert_close(model.observed[0], reference_model.observed[0], rtol=0, atol=0)
    # Three table entries acquire one float32 ULP through Wan's t/1000*1000 API.
    torch.testing.assert_close(model.observed[1], reference_model.observed[1].float(), rtol=1e-7, atol=0)
    torch.testing.assert_close(result.loss, expected_loss, rtol=0, atol=0)
    result.loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(model.scale.grad, reference_model.scale.grad, rtol=0, atol=0)
    if index in {416, 417, 418}:
        assert result.metrics["sigma"].item() == pytest.approx(0.876196146, rel=1e-7)


def test_reject_unknown_expert_and_changed_table_length(capability_class):
    with pytest.raises(ValueError, match="model.expert"):
        capability_class(TinyModel("both"))
    with pytest.raises(ValueError, match="num_train_timesteps"):
        capability_class(TinyModel(scheduler={"num_train_timesteps": 50}))
