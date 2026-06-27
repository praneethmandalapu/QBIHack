"""LangChain/LLM integration for clinical narrative generation."""


def generate_narrative(
    risk_score: float,
    shap_summary: dict,
    simulation_summary: dict | None = None,
) -> str:
    """Produce a plain-language summary of prediction and simulation results."""
    raise NotImplementedError
