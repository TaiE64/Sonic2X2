"""Merge LoRA factors back into plain weights (W' = W + BA*scale).

Turns an Any2Any training checkpoint (whose policy/value state dicts contain
LoRALinear structure: `<layer>.base.weight/.bias`, `<layer>.lora_A/B`) into a
stock SONIC-format checkpoint loadable by eval_agent_trl.py / deployment.

Usage:
  python merge_lora_checkpoint.py --ckpt <run_dir>/last.pt --out <merged.pt>
"""

from __future__ import annotations

import argparse

import torch


def merge_state_dict(sd: dict, alpha_over_rank: float | None = None) -> tuple[dict, int]:
    """Collapse every `X.base.weight` + `X.lora_A/B` triple into `X.weight`."""
    out = {}
    merged = 0
    lora_layers = {k[: -len(".lora_A")] for k in sd if k.endswith(".lora_A")}
    for k, v in sd.items():
        base = None
        for pfx in lora_layers:
            if k.startswith(pfx + "."):
                base = pfx
                break
        if base is None:
            out[k] = v
            continue
        suffix = k[len(base) + 1:]
        if suffix == "base.weight":
            A, B = sd[base + ".lora_A"], sd[base + ".lora_B"]
            rank = A.shape[0]
            scale = alpha_over_rank if alpha_over_rank is not None else 1.0
            out[base + ".weight"] = (v + (B @ A) * scale).to(v.dtype)
            merged += 1
        elif suffix == "base.bias":
            out[base + ".bias"] = v
        elif suffix in ("lora_A", "lora_B"):
            pass  # consumed above
        else:
            out[base + "." + suffix] = v
    return out, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha_over_rank", type=float, default=1.0,
                    help="LoRA scaling alpha/rank (we train with alpha=rank -> 1.0)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol, n_p = merge_state_dict(ck["policy_state_dict"], args.alpha_over_rank)
    val, n_v = (None, 0)
    if ck.get("value_state_dict") is not None:
        val, n_v = merge_state_dict(ck["value_state_dict"], args.alpha_over_rank)
    out = {"policy_state_dict": pol, "value_state_dict": val}
    if "state" in ck:  # eval_agent_trl.py reads checkpoint["state"].global_step
        out["state"] = ck["state"]
    torch.save(out, args.out)
    print(f"merged {n_p} actor + {n_v} critic LoRA layers -> {args.out}")

    # sanity: no lora keys remain; layer names match SONIC convention
    assert not any("lora_" in k or ".base." in k for k in pol), "unmerged keys!"
    probe = [k for k in pol if "g1_dyn.module.0.weight" in k]
    print("probe key:", probe, "shape:", pol[probe[0]].shape if probe else "?")


def self_test():
    import torch.nn as nn
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from lora import inject_lora
    torch.manual_seed(0)
    net = nn.ModuleDict({"dec": nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))})
    ref = {k: v.clone() for k, v in net.state_dict().items()}
    inject_lora(net, ["dec"], rank=2)
    # perturb lora_B so the merge is non-trivial
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.1)
    x = torch.randn(5, 8)
    y_wrapped = net["dec"](x)
    merged, n = merge_state_dict(net.state_dict(), 1.0)
    net2 = nn.ModuleDict({"dec": nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))})
    net2.load_state_dict(merged, strict=True)
    y_merged = net2["dec"](x)
    assert torch.allclose(y_wrapped, y_merged, atol=1e-6), "merge mismatch"
    assert n == 2
    print("merge_lora_checkpoint self_test: OK (wrapped == merged forward)")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
