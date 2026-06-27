"""Generate SHAP explanations for the trained risk model."""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "saved"


def explain(model_path: Path, features) -> dict:
    """Return SHAP values and summary for a patient feature vector."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Pass a model path and feature row to explain.")
