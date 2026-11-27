"""Paths and shared settings.

Values can come from environment variables or a git-ignored .env file in
the project root, using the same names as the team repo's .env.example:
DIGIKALA_DATA_DIR, QBC13_API_BASE_URL, QBC13_API_KEY.
"""

import os
from pathlib import Path

def _project_root() -> Path:
    """Folder with .env, data/, logs/: the nearest parent with pyproject.toml
    (team repo layout: src/digikala_assistant/comparison_analytics), otherwise
    the folder that contains this package (standalone layout)."""
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parent


ROOT = _project_root()


def env(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then the .env file, then the default."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    return default


RAW_DIR = Path(env("DIGIKALA_DATA_DIR", str(ROOT / "data")))
CLEAN_DIR = RAW_DIR / "clean"
API_BASE_URL = env("QBC13_API_BASE_URL", "https://193.93.169.12/v1")

# Minimum labelled comments before a product's percentages are trusted.
MIN_COMMENTS = 30
