import logging
import os
from datetime import datetime

def setup_logging(log_dir: str, log_name="experiment"):
    """初始化日志记录系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO
    )

    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info(f"Logging initialized. Log file: {log_path}")
    return logging.getLogger()
