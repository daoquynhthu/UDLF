from __future__ import annotations

import argparse
import json

import torch

from udlf.config import UDLFModelConfig
from udlf.model import UDLFStageAModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal UDLF stage A forward smoke check.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=31)
    parser.add_argument("--latent-slots", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--solver-steps", type=int, default=2)
    parser.add_argument("--diffusion-mode", choices=["ode", "fixed", "state_dependent"], default="ode")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = UDLFModelConfig(
        vocab_size=args.vocab_size,
        latent_slots=args.latent_slots,
        latent_dim=args.latent_dim,
        embed_dim=args.latent_dim,
        ff_multiplier=2,
        latent_heads=4,
        readout_heads=2,
        solver_steps=args.solver_steps,
        diffusion_mode=args.diffusion_mode,
    )
    model = UDLFStageAModel(config)
    input_ids = torch.randint(0, args.vocab_size, (args.batch, args.seq_len))
    generator = torch.Generator().manual_seed(args.seed)
    output = model(input_ids, generator=generator)
    result = {
        "batch": args.batch,
        "seq_len": args.seq_len,
        "logits_shape": list(output.logits.shape),
        "final_state_shape": list(output.final_state.shape),
        "loss": float(output.loss.detach().cpu()) if output.loss is not None else None,
        "diffusion_mode": args.diffusion_mode,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
