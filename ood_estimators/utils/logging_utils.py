import os
from pathlib import Path
import csv
from typing import Dict, List, Optional

from transformers import TrainerState

try:
    import comet_ml
    we_have_comet = True
except ImportError:
    print("comet_ml not found. If you install it using 'pip install comet_ml', "
          "you can also monitor the training process live.")
    we_have_comet = False

CALLBACKS_LOGGING_FNAME = "callbacks.tsv"


def log_callback(metrics: Dict[str, float], state: TrainerState, logging_dir: str):
    print(metrics)

    # if this is a first log, create a logging dir here
    Path(logging_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(logging_dir, CALLBACKS_LOGGING_FNAME), "a") as csv_file:
        logging_file = csv.writer(csv_file, delimiter="\t")

        for k, v in metrics.items():
            logging_file.writerow([state.global_step, state.epoch, k, v])

    if we_have_comet:
        experiment = comet_ml.config.get_global_experiment()
        if experiment is not None:
            experiment._log_metrics(metrics, step=state.global_step, epoch=state.epoch, framework="transformers")


def smoothed_vals(logs: List[float], smoothing_window: int) -> List[float]:
    smoothed_logs = []
    smooting_side_window = int(smoothing_window / 2)
    for i, val in enumerate(logs):
        lower = i-smooting_side_window if i-smooting_side_window > 0 else 0
        window_vals = logs[lower:i+smooting_side_window+1]
        window_mean = sum(window_vals) / len(window_vals)
        smoothed_logs.append(window_mean)
    return smoothed_logs


def best_val_idx(performance_logs: Dict[int, float],
                 picking_strategy: str,
                 early_stopping_patience: Optional[int] = 10,
                 smoothing_window: int = 0,
                 warmup: int = 50) -> int:
    logs_keys = list(performance_logs.keys())
    logs_vals = list(performance_logs.values())
    if smoothing_window:
        logs_vals = smoothed_vals(logs_vals, smoothing_window=smoothing_window)

    step_size = logs_keys[1] - logs_keys[0]
    best_idx = 0
    best_val = 100 if picking_strategy == "min" else -1
    for idx, val in zip(logs_keys, logs_vals):
        if idx < warmup:
            continue
        cond = val < best_val if picking_strategy == "min" else val > best_val
        # best match update
        if cond:
            best_val = val
            best_idx = idx
        # early stopping check
        if early_stopping_patience is not None and idx - best_idx >= early_stopping_patience*step_size:
            return best_idx

    # monotonous upgrading
    return best_idx
