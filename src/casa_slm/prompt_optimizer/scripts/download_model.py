"""Download the model from LitModels or Hugging Face Hub and save it to the specified cache path."""

import logging

from casa_slm.prompt_optimizer.config import load_config
from casa_slm.prompt_optimizer.lightning_utils import get_model_dir

logger = logging.getLogger(__name__)


def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    config = load_config()

    model_name = config.models.task_model.name
    hf_model_id = config.models.task_model.hf_model_id
    cache_path = config.models.task_model.path

    logger.info(f"Searching for {model_name}")
    model_dir = get_model_dir(model_name, hf_model_id, cache_path)
    logger.info(f"Model downloaded to: {model_dir}")


if __name__ == "__main__":
    main()
