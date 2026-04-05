"""
Shared utilities: config loading, logging, API client, GPU info.
All other modules import from here — do not duplicate these helpers.
"""

import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config and return as nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.
    Format: [YYYY-MM-DD HH:MM:SS] LEVEL name: message
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Loss CSV writer (for loss curve figures in report)
# ---------------------------------------------------------------------------

class LossCSVWriter:
    """Writes (step, loss) rows to a CSV file during training."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "loss"])

    def write(self, step: int, loss: float):
        self._writer.writerow([step, f"{loss:.6f}"])
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def log_gpu_info(logger: logging.Logger) -> None:
    """
    Log GPU name, memory, and driver version via nvidia-smi.
    Satisfies the 'GPU allocation' grading evidence requirement.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for i, line in enumerate(result.stdout.strip().splitlines()):
            logger.info(f"GPU {i}: {line.strip()}")
    except Exception as e:
        logger.warning(f"nvidia-smi unavailable: {e}")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def load_api_client() -> OpenAI:
    """
    Build an OpenAI client pointed at the UTSA endpoint.
    Reads OPENAI_API_KEY and UTSA_BASE_URL from .env (or environment).
    """
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("UTSA_BASE_URL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — check your .env file")
    if not base_url:
        raise RuntimeError("UTSA_BASE_URL not set — check your .env file")

    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_name(config: dict, role: str) -> str:
    """
    Return the model name for 'teacher' or 'judge'.
    UTSA_MODEL env var overrides config if set (useful for quick swaps).
    """
    load_dotenv()
    env_override = os.environ.get("UTSA_MODEL")
    if env_override and role in ("teacher", "judge"):
        return env_override
    return config["model"][role]
