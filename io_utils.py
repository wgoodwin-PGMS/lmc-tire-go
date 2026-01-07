import os
import logging
import shutil
import random

import psutil
import joblib
import orjson

from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def get_json_payload(path: str) -> dict:  # Return dict, not str
    with open(path, "rb") as f:  # Binary mode
        return orjson.loads(f.read())


def save_checkpoint(optimizer, checkpoint_dir: str, generation: int):
    """Save optimizer state to checkpoint file"""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_gen_{generation}.pkl"

    checkpoint_data = {
        "generation": optimizer.generation,
        "population": optimizer.population,
        "ref_dirs": optimizer.ref_dirs,
        "f_x": optimizer.f_x,
        "n_objectives": optimizer.n_objectives,
        "ref_point": optimizer.convergence_tracker.ref_point,
        "convergence_history": optimizer.convergence_tracker.history,
        "seen_individuals": optimizer._ga_population._seen,
        "removed_objectives": optimizer.ga_analysis.removed_objectives,
        "seed_state": {"numpy": np.random.get_state(), "random": random.getstate()},
    }

    joblib.dump(checkpoint_data, checkpoint_path)
    logger.info(f"    [RESTART] Checkpoint saved to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path: str):
    """Load optimizer state from checkpoint file"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_data = joblib.load(checkpoint_path)
    logger.info(f"    [RESTART] Checkpoint loaded from {checkpoint_path}")

    return checkpoint_data


def free_storage():
    stat = shutil.disk_usage("/")
    output = f"Used {round(100*stat[1]/stat[0], 2)}% of Disk Space"

    return output


def mem_usage():
    process = psutil.Process(os.getpid())

    # Get memory usage of this process (RSS = Resident Set Size)
    rss = process.memory_info().rss  # in bytes

    # Convert to MB
    rss_mb = rss / 1024**2

    # Get total system memory
    total_mb = psutil.virtual_memory().total / 1024**2

    # Percentage
    percent = (rss / psutil.virtual_memory().total) * 100

    output = f"Memory usage: {rss_mb:.2f} MB ({percent:.2f}% of total system memory)"

    return output
