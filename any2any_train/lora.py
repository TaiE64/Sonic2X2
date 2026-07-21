"""LoRA for Any2Any dynamics adaptation (paper Eq. 9).

    W' = W + B A,   A in R^{k x d_in},  B in R^{d_out x k},  only {A, B} trained.

Faithful-to-paper details:
  - B is zero-initialized so that at step 0 the adapted policy is exactly the
    frozen source policy (standard LoRA init; the paper does not specify, we
    record this as the documented default — repro plan gap #1).
  - Injection sites for the Sonic backbone (paper Sec. 4.2, "Sonic as source"):
    every nn.Linear inside the actor's dynamics decoder and the critic network.
    The FSQ bottleneck, the Robot Motion Encoder, and all other pretrained
    components stay frozen (repro plan gap #2: exact linear subset unspecified;
    default = all nn.Linear under the targeted submodules, and we print the
    trainable-parameter fraction for the ~5% sanity check).

Works on torch only; no Isaac Lab imports, so it is unit-testable anywhere.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement wrapping a frozen nn.Linear with a low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float | None = None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / rank
        # A: kaiming-uniform (as in the reference LoRA implementation); B: zeros.
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B) * self.scaling

    def extra_repr(self) -> str:
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}")


def inject_lora(root: nn.Module, target_submodules: Iterable[str],
                rank: int = 16, alpha: float | None = None,
                exclude_suffixes: Iterable[str] | None = None) -> list[str]:
    """Replace every nn.Linear under the named submodules with LoRALinear.

    `target_submodules` are attribute paths relative to `root`
    (e.g. ["actor.decoder", "critic"]). Returns the qualified names of all
    wrapped linears. Everything else in `root` is frozen.

    `exclude_suffixes`: qualified-name suffixes to skip (e.g. ".module.0",
    ".module.12" for the strict-FFN-only variant that leaves the actor's input
    projection and action head frozen, matching the paper Fig.2 W_FFN drawing).
    """
    exclude = tuple(exclude_suffixes or ())
    # freeze the whole backbone first
    for p in root.parameters():
        p.requires_grad_(False)

    wrapped: list[str] = []
    skipped: list[str] = []
    for sub_path in target_submodules:
        sub = root.get_submodule(sub_path)
        if isinstance(sub, nn.Linear):  # the target itself is a linear
            parent_path, _, leaf = sub_path.rpartition(".")
            parent = root.get_submodule(parent_path) if parent_path else root
            setattr(parent, leaf, LoRALinear(sub, rank, alpha))
            wrapped.append(sub_path)
            continue
        for name, mod in list(sub.named_modules()):
            if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
                qname = f"{sub_path}.{name}"
                if any(qname.endswith(s) for s in exclude):
                    skipped.append(qname)
                    continue
                parent_path, _, leaf = name.rpartition(".")
                parent = sub.get_submodule(parent_path) if parent_path else sub
                setattr(parent, leaf, LoRALinear(mod, rank, alpha))
                wrapped.append(qname)
    if not wrapped:
        raise ValueError(f"no nn.Linear found under {list(target_submodules)}")
    if skipped:
        print(f"[inject_lora] excluded {len(skipped)} linears: {skipped}")  # noqa: T201
    return wrapped


def lora_parameters(root: nn.Module):
    """The trainable {A, B} parameters (for optimizer param groups)."""
    for mod in root.modules():
        if isinstance(mod, LoRALinear):
            yield mod.lora_A
            yield mod.lora_B


def trainable_fraction(root: nn.Module) -> float:
    tot = sum(p.numel() for p in root.parameters())
    trn = sum(p.numel() for p in root.parameters() if p.requires_grad)
    return trn / max(tot, 1)


# ---------------------------------------------------------------------------
# Freeze-correctness checks (repro plan Sec. 3.4 "冻结校验单测")
# ---------------------------------------------------------------------------

def check_optimizer_only_lora(optimizer: torch.optim.Optimizer, root: nn.Module):
    """Every tensor the optimizer updates must be a LoRA A/B."""
    lora_ids = {id(p) for p in lora_parameters(root)}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in lora_ids:
                raise AssertionError("optimizer contains a non-LoRA parameter")
    return True


def check_frozen_unchanged(root: nn.Module, step_fn) -> bool:
    """Run one optimization step via step_fn(); frozen weights must not move."""
    before = {n: p.detach().clone() for n, p in root.named_parameters()
              if not p.requires_grad}
    step_fn()
    for n, p in root.named_parameters():
        if not p.requires_grad and not torch.equal(before[n], p.detach()):
            raise AssertionError(f"frozen parameter changed: {n}")
    return True


def check_identity_at_init(root: nn.Module, root_ref: nn.Module,
                           example_input) -> bool:
    """With B=0 the wrapped model must output exactly what the source does."""
    root.eval(); root_ref.eval()
    with torch.no_grad():
        out = root(example_input)
        ref = root_ref(example_input)
    if not torch.allclose(out, ref, atol=0, rtol=0):
        raise AssertionError("LoRA-injected model differs from source at init "
                             "(B=0 identity property violated)")
    return True
