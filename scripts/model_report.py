from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pidmppo.config import load_config
from pidmppo.experiment import VARIANTS, apply_variant
from pidmppo.models import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Report parameter counts for baseline fairness")
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/model_parameters.csv"))
    args = parser.parse_args()
    base = load_config(args.config)
    rows = []
    for variant in VARIANTS:
        config = apply_variant(base, variant)
        model = ActorCritic(config.env.observation_dim, 2, config.model)
        rows.append(
            {
                "variant": variant,
                "encoder": config.model.encoder,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "dual_value_heads": config.model.twin_critic,
                "l_head": config.model.auxiliary_heads and config.model.enable_l_head,
                "g_head": config.model.auxiliary_heads and config.model.enable_g_head,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"variants={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
