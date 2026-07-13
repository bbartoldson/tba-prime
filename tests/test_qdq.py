import pytest
import torch

from zeroband.training.qdq import E2M1_MAX, nvfp4_qdq, nvfp4_qdq_ste, should_quantize_key


def test_grid_values_are_fixed_points():
    # Values already on the grid (with a power-of-two block scale) must survive QDQ.
    scale = 0.25  # exactly representable in E4M3
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) * scale
    t = torch.cat([grid, grid]).unsqueeze(0)  # one block of 16
    out = nvfp4_qdq(t.to(torch.float32))
    assert torch.equal(out, t)


def test_hand_computed_rounding():
    # Block amax = 6 -> scale = 1 (exact in E4M3), so values round on the raw grid.
    vals = [6.0, 5.2, 4.9, 2.4, 2.6, 0.2, 0.3, -0.7, -1.3, 1.6, 3.4, 3.6, -4.8, 0.74, 0.76, 1.0]
    expect = [6.0, 6.0, 4.0, 2.0, 3.0, 0.0, 0.5, -0.5, -1.5, 1.5, 3.0, 4.0, -4.0, 0.5, 1.0, 1.0]
    out = nvfp4_qdq(torch.tensor(vals, dtype=torch.float32).unsqueeze(0))
    assert torch.equal(out.squeeze(0), torch.tensor(expect))


def test_idempotent():
    t = torch.randn(64, 128, dtype=torch.bfloat16)
    once = nvfp4_qdq(t)
    twice = nvfp4_qdq(once)
    assert torch.equal(once, twice)


def test_error_bounded():
    t = torch.randn(64, 128, dtype=torch.float32)
    out = nvfp4_qdq(t)
    blocks = t.reshape(-1, 16)
    # Worst-case relative step on the E2M1 grid is 1/6 of block amax (gap 5->6
    # is 1, plus E4M3 scale rounding ~2^-3 relative).
    bound = blocks.abs().amax(dim=-1, keepdim=True) / E2M1_MAX * (1.0 + 0.13)
    err = (out.reshape(-1, 16) - blocks).abs()
    assert (err <= bound + 1e-6).all()


def test_padding_last_dim_not_multiple_of_block():
    t = torch.randn(4, 30, dtype=torch.float32)
    out = nvfp4_qdq(t)
    assert out.shape == t.shape
    # Padded block must not perturb the values that share it.
    assert torch.equal(out[:, 16:], nvfp4_qdq(torch.nn.functional.pad(t[:, 16:], (0, 2)))[:, :14])


def test_zero_block():
    t = torch.zeros(2, 16)
    assert torch.equal(nvfp4_qdq(t), t)


def test_global_tensor_scale_close_to_local():
    t = torch.randn(64, 128, dtype=torch.float32)
    local = nvfp4_qdq(t, global_scale_mode="none")
    glob = nvfp4_qdq(t, global_scale_mode="tensor")
    # Same grid resolution regime -> errors within 2x of each other on average.
    e_local = (local - t).abs().mean()
    e_glob = (glob - t).abs().mean()
    assert e_glob < 2 * e_local and e_local < 2 * e_glob


def test_four_over_six_never_worse():
    t = torch.randn(256, 64, dtype=torch.float32)
    base = nvfp4_qdq(t, four_over_six=False)
    fos = nvfp4_qdq(t, four_over_six=True)
    mse = lambda a: (a - t).pow(2).mean()
    assert mse(fos) <= mse(base) + 1e-12


def test_ste_gradient_is_identity():
    t = torch.randn(8, 32, requires_grad=True)
    nvfp4_qdq_ste(t).sum().backward()
    assert torch.equal(t.grad, torch.ones_like(t))


def test_bf16_deterministic():
    t = torch.randn(32, 64, dtype=torch.bfloat16)
    assert torch.equal(nvfp4_qdq(t), nvfp4_qdq(t.clone()))


@pytest.mark.parametrize(
    "key,ndim,expected",
    [
        ("model.layers.0.self_attn.q_proj.weight", 2, True),
        ("model.layers.3.mlp.gate_proj.weight", 2, True),
        ("model.embed_tokens.weight", 2, False),
        ("lm_head.weight", 2, False),
        ("model.layers.0.input_layernorm.weight", 1, False),
        ("model.layers.0.self_attn.q_proj.bias", 1, False),
    ],
)
def test_should_quantize_key(key, ndim, expected):
    assert should_quantize_key(key, ndim) == expected


def test_skip_last_frac():
    assert should_quantize_key("model.layers.35.mlp.up_proj.weight", 2, num_layers=36, skip_last_frac=0.15) is False
    assert should_quantize_key("model.layers.10.mlp.up_proj.weight", 2, num_layers=36, skip_last_frac=0.15) is True


def test_apply_fake_quant_forward():
    from types import SimpleNamespace

    from zeroband.training.qdq import apply_fake_quant_forward

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=2)
            self.layers = torch.nn.ModuleList([
                torch.nn.ModuleDict({"q_proj": torch.nn.Linear(32, 32, bias=False)}) for _ in range(2)
            ])
            self.lm_head = torch.nn.Linear(32, 8, bias=False)

        def forward(self, x):
            for layer in self.layers:
                x = layer["q_proj"](x)
            return self.lm_head(x)

    model = Tiny()
    n = apply_fake_quant_forward(model, "nvfp4")
    assert n == 2  # lm_head skipped

    # Patched forward must compute with the QDQ'd weight...
    x = torch.randn(4, 32)
    w = model.layers[0]["q_proj"].weight
    expected = torch.nn.functional.linear(x, nvfp4_qdq(w))
    assert torch.equal(model.layers[0]["q_proj"](x), expected)

    # ...and STE must pass gradients through to the raw weight.
    model(x).sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0
    assert model.lm_head.weight.grad is not None
