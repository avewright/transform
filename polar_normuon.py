"""Polar Express orthogonalization for NorMuon (modded-nanogpt style).

Portable pure-PyTorch version of:
  https://github.com/KellerJordan/modded-nanogpt (Polar Express + NorMuon)
  Paper: https://arxiv.org/pdf/2505.16932

Replaces Newton–Schulz in the pip `normuon` package without requiring Triton/FA3/FP8.
"""
from __future__ import annotations

import torch

# Computed for num_iters=5, safety_factor=2e-2, cushion=2 (modded-nanogpt)
_POLAR_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)

_polar_fn = None


def _polar_express_impl(G: torch.Tensor) -> torch.Tensor:
    """Zeroth-power / orthogonalization via Polar Express (bf16-stable)."""
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    # Spectral norm ≤ 1 with safety factor (modded-nanogpt)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    for a, b, c in _POLAR_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def get_polar_express(*, compile_polar: bool = True):
    """Return (possibly compiled) Polar Express fn. Cached."""
    global _polar_fn
    if _polar_fn is not None:
        return _polar_fn
    fn = _polar_express_impl
    if compile_polar and hasattr(torch, "compile"):
        try:
            fn = torch.compile(fn, dynamic=False, fullgraph=True)
        except Exception:
            pass
    _polar_fn = fn
    return _polar_fn


def normuon_update_polar(
    grad,
    momentum,
    second_momentum,
    beta=0.95,
    beta2=0.95,
    nesterov=True,
    *,
    compile_polar: bool = True,
):
    """NorMuon update with Polar Express instead of Newton–Schulz."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    polar = get_polar_express(compile_polar=compile_polar)
    update = polar(update).to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # NorMuon variance reduction (same as pip normuon)
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceNorMuonPolarWithAuxAdam(torch.optim.Optimizer):
    """Single-device NorMuon+Adam with Polar Express + optional cautious WD.

    API mirrors `normuon.SingleDeviceNorMuonWithAuxAdam` param groups.
    """

    def __init__(self, param_groups, *, cautious_wd: bool = True, compile_polar: bool = True):
        self.cautious_wd = bool(cautious_wd)
        self.compile_polar = bool(compile_polar)
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params", "lr", "momentum", "beta2", "weight_decay", "use_muon",
                }
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params", "lr", "betas", "eps", "weight_decay", "use_muon",
                }
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update_polar(
                        p.grad,
                        state["momentum_buffer"],
                        state["second_momentum_buffer"],
                        beta=group["momentum"],
                        beta2=group["beta2"],
                        compile_polar=self.compile_polar,
                    )
                    lr = group["lr"]
                    wd = group["weight_decay"]
                    if wd and had_grad:
                        if self.cautious_wd:
                            # Decay only where update and param share sign (modded-nanogpt).
                            mask = (update.reshape(p.shape) * p) >= 0
                            p.sub_(p * mask.to(p.dtype) * (lr * wd))
                        else:
                            p.mul_(1 - lr * wd)
                    p.add_(update.reshape(p.shape), alpha=-lr)
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"],
                    )
                    lr = group["lr"]
                    wd = group["weight_decay"]
                    if wd and had_grad:
                        if self.cautious_wd:
                            mask = (update * p) >= 0
                            p.sub_(p * mask.to(p.dtype) * (lr * wd))
                        else:
                            p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
        return loss


def unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)
