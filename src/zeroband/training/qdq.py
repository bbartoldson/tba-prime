"""NVFP4 quantize-dequantize (QDQ) emulation.

Emulates NVFP4 numerics (E2M1 values with per-16-block E4M3 scales) in BF16
compute so that quantization *error* is reproduced exactly without FP4
hardware. Applying the same function on the trainer (fake-quant forward) and
the sampler (rollout checkpoint) gives a bit-exact trainer/sampler contract by
construction.

Deviations from true NVFP4, chosen deliberately:
- Default ``global_scale_mode="none"`` omits the tensor-level FP32 scale so
  the QDQ is purely block-local. This keeps trainer-side fake-quant exact
  under FSDP sharding (no cross-rank amax reduction needed). For BF16-scale
  LLM weights the E4M3 block-scale range is nowhere near saturated, so the
  error profile matches true NVFP4. ``global_scale_mode="tensor"`` implements
  the full spec for comparison.
- Activations are not quantized (weight-only emulation, W4A16-like).
"""

import re

import torch

# E2M1 representable magnitudes.
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between adjacent grid values; ties resolve to the lower magnitude
# (deterministic; exact ties are measure-zero for trained weights).
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

_DEVICE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _grid_for(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if device not in _DEVICE_CACHE:
        _DEVICE_CACHE[device] = (_E2M1_GRID.to(device), _E2M1_MIDPOINTS.to(device))
    return _DEVICE_CACHE[device]

E2M1_MAX = 6.0
E4M3_MAX = 448.0


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round values (already scaled into [-6, 6]) to the E2M1 grid."""
    grid, midpoints = _grid_for(x.device)
    sign = torch.sign(x)
    mag = x.abs().clamp(max=E2M1_MAX)
    idx = torch.bucketize(mag, midpoints, right=True)
    return sign * grid[idx]


def _e4m3_roundtrip(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def nvfp4_qdq(
    t: torch.Tensor,
    block_size: int = 16,
    global_scale_mode: str = "none",
    four_over_six: bool = False,
) -> torch.Tensor:
    """Quantize-dequantize ``t`` to NVFP4 numerics along the last dim.

    Returns a tensor of the same shape and dtype whose values lie on the
    NVFP4-representable grid.
    """
    assert global_scale_mode in ("none", "tensor"), global_scale_mode
    orig_dtype = t.dtype
    orig_shape = t.shape
    x = t.detach().to(torch.float32)

    k = orig_shape[-1]
    pad = (-k) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x.reshape(-1, block_size)

    if global_scale_mode == "tensor":
        tensor_amax = x.abs().max()
        global_scale = (tensor_amax / (E2M1_MAX * E4M3_MAX)).clamp(min=1e-30)
    else:
        global_scale = x.new_tensor(1.0)

    block_amax = x.abs().amax(dim=-1, keepdim=True)

    def _dq_for(divisor: float) -> torch.Tensor:
        scale = _e4m3_roundtrip(block_amax / (divisor * global_scale))
        full_scale = scale * global_scale
        safe = torch.where(full_scale > 0, full_scale, torch.ones_like(full_scale))
        q = _round_e2m1(x / safe)
        return torch.where(full_scale > 0, q * full_scale, torch.zeros_like(x))

    dq = _dq_for(E2M1_MAX)
    if four_over_six:
        dq4 = _dq_for(4.0)
        err6 = (dq - x).pow(2).sum(dim=-1, keepdim=True)
        err4 = (dq4 - x).pow(2).sum(dim=-1, keepdim=True)
        dq = torch.where(err4 < err6, dq4, dq)

    dq = dq.reshape(*orig_shape[:-1], k + pad)
    if pad:
        dq = dq[..., :k]
    return dq.to(orig_dtype)


def nvfp4_qdq_ste(t: torch.Tensor, **kwargs) -> torch.Tensor:
    """QDQ with straight-through gradient (for fake-quant training forwards)."""
    return t + (nvfp4_qdq(t, **kwargs) - t).detach()


# Weights whose state-dict key matches any of these substrings stay in BF16
# (mirrors the humans& / NVIDIA selective-precision convention: quantize only
# the big linear projections).
_SKIP_SUBSTRINGS = ("embed", "lm_head", "norm", "bias")
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def should_quantize_key(key: str, ndim: int, num_layers: int | None = None, skip_last_frac: float = 0.0) -> bool:
    """Decide whether a state-dict entry participates in QDQ."""
    if ndim != 2 or not key.endswith(".weight"):
        return False
    if any(s in key for s in _SKIP_SUBSTRINGS):
        return False
    if skip_last_frac > 0.0 and num_layers is not None:
        m = _LAYER_RE.search(key)
        if m is not None and int(m.group(1)) >= num_layers * (1.0 - skip_last_frac):
            return False
    return True


def apply_fake_quant_forward(
    model: torch.nn.Module,
    mode: str,
    four_over_six: bool = False,
    skip_last_frac: float = 0.0,
) -> int:
    """Monkeypatch eligible ``nn.Linear`` forwards to QDQ their weight (STE).

    Reads ``self.weight`` at call time, so it composes with FSDP2 (which
    unshards parameters in its pre-forward hook) and leaves the state dict
    untouched (checkpointing / rollout save see the raw BF16 weights).

    Returns the number of patched modules.
    """
    import types

    assert mode == "nvfp4", f"unknown fake_quant_forward mode {mode}"
    num_layers = getattr(model.config, "num_hidden_layers", None)

    def _make_forward(fos: bool):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            weight = nvfp4_qdq_ste(self.weight, four_over_six=fos)
            return torch.nn.functional.linear(input, weight, self.bias)

        return forward

    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and should_quantize_key(
            name + ".weight", module.weight.ndim, num_layers=num_layers, skip_last_frac=skip_last_frac
        ):
            module.forward = types.MethodType(_make_forward(four_over_six), module)
            patched += 1
    return patched
