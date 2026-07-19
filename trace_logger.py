import logging
import os
from datetime import datetime

_logger = None


def init_trace_logger(model_name: str, num_runs: int, mode: str, log_dir: str = "logs") -> str:
    """Initialise the trace logger and return the log file path."""
    global _logger
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    safe_model = model_name.replace("/", "_").replace(":", "_").replace(".", "_")
    filename = f"{safe_model}_{num_runs}_{mode}_{date_str}.log"
    filepath = os.path.join(log_dir, filename)

    logger = logging.getLogger("mdt_trace")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(filepath, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _logger = logger
    _logger.info(f"=== TRACE START | model={model_name} | runs={num_runs} | mode={mode} ===")
    return filepath


def get_trace_logger():
    if _logger is None:
        raise RuntimeError("Trace logger not initialised. Call init_trace_logger() first.")
    return _logger
