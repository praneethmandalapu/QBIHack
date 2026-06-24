import os
from pathlib import Path

# Base Directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
DEMO_DIR = DATA_DIR / "demo"
CACHE_DIR = DATA_DIR / "cache"
NETWORKS_DIR = DATA_DIR / "networks"
MODELS_DIR = PROJECT_ROOT / "models"

# Supported cancer types
SUPPORTED_CANCERS = ["KIRC", "LIHC"]

# Risk probability thresholds
RISK_LOW_THRESHOLD = 0.35
RISK_HIGH_THRESHOLD = 0.65

# Model training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# External API endpoints
DGIDB_API_URL = os.getenv("DGIDB_API_URL", "https://dgidb.org/api/v2/interactions.json")

# Debug mode
DEBUG = os.getenv("ONCOPULSE_DEBUG", "false").lower() in ("true", "1", "yes")

# Curated CPG and Pathway Genes derived from paper
KIRC_GENES = [
    "DNMT3B", "PPP1R1A",  # Key unfavourable regulators
    "CLDN1", "CLDN2", "CLDN3", "CLDN4", "CLDN7", "CLDN8",  # Claudins (favourable)
    "OCLN", "TJP1", "TJP2",  # Occludin, Zonula Occludens
    "F11R", "JAM2", "JAM3", "CGN", "MPP5", "PARD3", "PARD6A"  # Other tight junction genes
]

LIHC_GENES = [
    "TAF15", "CHEK1", "PDCD6",  # Key regulators with highest slope values in both pathways
    "PRPS1", "PRPS2", "ADSL", "IMPDH1", "IMPDH2", "XDH", "NME1", "NME2",  # Purine metabolism genes
    "POLR1A", "POLR2A", "POLR3A"  # RNA polymerase genes
]

# Ensure required directories exist
for path in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, 
    DEMO_DIR, CACHE_DIR, NETWORKS_DIR, MODELS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)
