"""Color maps for necrotic, viable, and healthy tissue regions."""

NECROTIC = "#8B0000"
VIABLE = "#FF4500"
HEALTHY = "#90EE90"


def tissue_colormap():
    """Return a colormap dict or palette for visualization."""
    return {"necrotic": NECROTIC, "viable": VIABLE, "healthy": HEALTHY}
