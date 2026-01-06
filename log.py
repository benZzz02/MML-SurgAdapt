import logging
import os
import sys

from config import cfg

cfg.log_file = f"{cfg.checkpoint}/{cfg.log_file}"

rank = int(os.environ.get("RANK", "0"))
handlers = []
if rank == 0:
    handlers = [
        logging.FileHandler(cfg.log_file),
        logging.StreamHandler(sys.stdout)
    ]
else:
    handlers = [logging.NullHandler()]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=handlers)

logger = logging.getLogger()

if cfg.resume and rank == 0:
    logger.info(f"Resume training... from {cfg.resume}")
