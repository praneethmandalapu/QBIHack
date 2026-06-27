"""Train XGBoost risk model on METABRIC genomic/clinical features."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent / "saved"


def train() -> Path:
    """Load processed features, fit model, and save to models-praneeth/saved/."""
    raise NotImplementedError


if __name__ == "__main__":
    train()
