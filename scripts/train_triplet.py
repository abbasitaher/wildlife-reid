from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildlife_reid.config import load_config
from wildlife_reid.training.train_triplet import train_triplet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a triplet-loss embedding model.")
    parser.add_argument("--config", default="configs/sea_turtle.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = train_triplet(config)
    print(f"Best checkpoint saved to {checkpoint}")


if __name__ == "__main__":
    main()
