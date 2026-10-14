# Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Common utility functions for the Lightning platform."""

import logging
import os
from pathlib import Path

import huggingface_hub as hf
import litmodels
from dotenv import load_dotenv

__all__ = ["get_lightning_model_base_folder", "get_model_dir", "is_running_in_lightning"]

logger = logging.getLogger(__name__)

_LIGHTNING_DRIVE = Path("/teamspace/lightning_storage/shared-models/")
_LITMODEL_PREFIX = "outshift-cisco/casa/"


def is_running_in_lightning() -> bool:
    """Check if the current environment is a Ligthning Studio."""
    return "LIGHTNING_USERNAME" in os.environ


def get_lightning_model_base_folder() -> Path:
    """Get the directory in which the models are stored."""
    return _LIGHTNING_DRIVE


def get_model_dir(model_name: str, hf_model_id: str | None = None, cache_path: Path | None = None) -> Path:
    """Get the directory for a model, downloading it if necessary.

    Try to download the model from LitModels first. If that fails, fall back to downloading from Hugging Face Hub.

    Args:
        model_name (str): The name of the model.
        hf_model_id (str | None): The Hugging Face model ID.
        cache_path (Path | None): The path to cache the model.

    Raises:
        RuntimeError: If the model cannot be downloaded from either LitModels or Hugging Face Hub.

    Returns:
        Path: The path to the model directory.
    """
    load_dotenv()

    if cache_path is not None:
        model_path = cache_path / model_name
    else:
        model_path = get_lightning_model_base_folder() / model_name

    model_path.mkdir(parents=True, exist_ok=True)

    litmodel_name = _LITMODEL_PREFIX + model_name

    # Download the model from LitModels. If it fails, fall back to downloading from Hugging Face Hub.
    try:
        logger.info(f"Searching LitModels for {litmodel_name}")
        litmodels.download_model(litmodel_name, model_path)
        return model_path
    except Exception as e:
        logger.error(f"Model does not exist on LitModels: {e}")

    try:
        hf_token = os.getenv("HF_TOKEN", "")
        if not hf_token:
            logger.warning("HF_TOKEN is not set.")

        if hf_model_id is None:
            raise ValueError("hf_model_id must be provided if the model is not available on LitModels.")

        logger.info(f"Downloading {hf_model_id} to local storge.")
        hf.snapshot_download(repo_id=hf_model_id, local_dir=model_path)

        # Upload the model to LitModels for future use
        logger.info(f"Uploading {litmodel_name} to LitModels")
        litmodels.upload_model(litmodel_name, model_path)
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model from Hugging Face Hub: {e}")
        raise RuntimeError(f"Failed to download model {model_name} from both LitModels and Hugging Face Hub.")
