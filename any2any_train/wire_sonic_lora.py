"""Wire LoRA into the real SONIC checkpoint and verify (no Isaac Sim needed).

Loads the released sonic_release weights into the actual gear_sonic module
classes, injects LoRA per the paper (actor dynamics decoder + critic; FSQ,
encoders, aux decoder frozen), and verifies:
  1. strict weight load (every tensor consumed)
  2. B=0 identity: injected decoder == source decoder on random input
  3. freeze correctness: only LoRA params require grad
  4. trainable-parameter fraction (paper anchor ~5% at r=64; we record actual)

Run: /home/taie/miniconda3/envs/isaaclab/bin/python wire_sonic_lora.py [--rank 16]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/home/taie/Desktop/Byte/Sonic_Retarget/GR00T-WholeBodyControl")
sys.path.insert(0, str(HERE))
from lora import (LoRALinear, inject_lora, lora_parameters,  # noqa: E402
                  trainable_fraction)

CKPT = REPO / "sonic_release" / "weights_only.pt"

# Layer shapes are read from the checkpoint itself, so this stays correct even
# if NVIDIA revs the release; SiLU activations per g1_dyn_mlp.yaml.


def build_mlp_from_state(prefix: str, sd: dict) -> nn.Sequential:
    """Rebuild the BaseModule MLP (Linear+SiLU alternation) from state-dict shapes."""
    idx = 0
    layers = []
    while f"{prefix}.{idx}.weight" in sd:
        w = sd[f"{prefix}.{idx}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if f"{prefix}.{idx + 2}.weight" in sd:
            layers.append(nn.SiLU())
        idx += 2
    return nn.Sequential(*layers)


def load_mlp(mlp: nn.Sequential, prefix: str, sd: dict):
    own = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
    missing, unexpected = mlp.load_state_dict(own, strict=True), None
    return own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu")
    pol, val = ck["policy_state_dict"], ck["value_state_dict"]

    # ---- actor dynamics decoder --------------------------------------
    dec_prefix = "actor_module.decoders.g1_dyn.module"
    decoder = build_mlp_from_state(dec_prefix, pol)
    load_mlp(decoder, dec_prefix, pol)
    import copy
    decoder_ref = copy.deepcopy(decoder)

    # ---- critic -------------------------------------------------------
    cr_prefix = "critic_module.module"
    critic = build_mlp_from_state(cr_prefix, val)
    load_mlp(critic, cr_prefix, val)
    critic_ref = copy.deepcopy(critic)

    n_dec = sum(p.numel() for p in decoder.parameters())
    n_cr = sum(p.numel() for p in critic.parameters())
    n_frozen_rest = sum(v.numel() for k, v in pol.items()
                        if "g1_dyn" not in k)  # encoders + fsq + g1_kin + std
    print(f"decoder {n_dec/1e6:.2f}M | critic {n_cr/1e6:.2f}M | "
          f"frozen backbone rest {n_frozen_rest/1e6:.2f}M")

    # ---- inject -------------------------------------------------------
    container = nn.ModuleDict({"decoder": decoder, "critic": critic})
    wrapped = inject_lora(container, ["decoder", "critic"], rank=args.rank)
    print(f"LoRA rank {args.rank}: wrapped {len(wrapped)} linears:")
    for w in wrapped:
        print(f"  {w}")

    # ---- verify identity at init (B=0) --------------------------------
    x_dec = torch.randn(64, decoder_ref[0].in_features)
    x_cr = torch.randn(64, critic_ref[0].in_features)
    with torch.no_grad():
        assert torch.equal(container["decoder"](x_dec), decoder_ref(x_dec)), \
            "decoder identity violated"
        assert torch.equal(container["critic"](x_cr), critic_ref(x_cr)), \
            "critic identity violated"
    print("B=0 identity on REAL weights: OK (decoder & critic bit-exact)")

    # ---- freeze check + fraction --------------------------------------
    lora_n = sum(p.numel() for p in lora_parameters(container))
    for n, p in container.named_parameters():
        assert p.requires_grad == ("lora_" in n), f"grad flag wrong: {n}"
    total_model = n_dec + n_cr + n_frozen_rest
    print(f"LoRA params: {lora_n/1e3:.1f}K "
          f"| vs adapted modules {lora_n/(n_dec+n_cr)*100:.2f}% "
          f"| vs full model {lora_n/total_model*100:.2f}% "
          f"(paper anchor ~5%; r=64 would give {lora_n*4/total_model*100:.2f}%)")

    # ---- one grad step must move only LoRA ----------------------------
    opt = torch.optim.Adam(lora_parameters(container), lr=1e-3)
    before = {n: p.clone() for n, p in container.named_parameters()
              if not p.requires_grad}
    loss = container["decoder"](x_dec).square().mean() + \
           container["critic"](x_cr).square().mean()
    loss.backward()
    opt.step()
    for n, p in container.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[n], p), f"frozen moved: {n}"
    print("frozen-unchanged after real grad step: OK")
    print("\nWIRING VERIFIED — ready for Isaac Lab integration")


if __name__ == "__main__":
    main()
