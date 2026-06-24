import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    """Helper to get a named logger."""
    return logging.getLogger(name)

def get_current_timestamp() -> str:
    """Get formatting timestamp."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
