"""
canarydecoder.decoding._models
------------------------------

Pre-trained models indexation.
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np

import reservoirpy

# path to the pretrained models. Should be a remote repository if
# this code is released one day.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

# all available models.
MODEL_LIST = [
    "canary16",
    "canary16-deltas",
    "canarygan-1120",
    "canarygan-2255",
    "canarygan-2255-d",
    "canarygan-1120-d",
    "canarygan-3",
    "canarygan-3-d",
    "canarygan-e-ot-noise",
    "canary16-clean-d",
    "canary16-clean-d-notrim",
    "canarygan-clean-e-ot-noise-notrim",
    "canarygan-clean-e-ot-noise",
    "canary16-filtered",
    "canary16-filtered-notrim",
    "canarygan-f-3e-ot-noise",
    "canarygan-f-3e-ot-noise-notrim",
    "canarygan-8e-ot-noise",
    "canarygan-2e-ot-noise",
    "canarygan-8e-ot-noise-v2",
    "021220-1e",
    "021220-1e-balanced",
    "021220-8e",
    "021220-8e-balanced"
]


def _get_model(name: str):
    """Get model by name.

    Arguments:
    ----------
        name {str} -- Name of the model.

    Raises:
    -------
        ValueError: Bad name or unavailable model.

    Returns:
    --------
        Model -- Loaded pre-trained model.
    """

    # retrieve known model
    if name in MODEL_LIST or os.path.exists(name):
        if name in MODEL_LIST:
            model_dir = Path(MODEL_PATH, name)
        else:
            model_dir = Path(name)

        # 1st case: this a reservoirpy v0.3 model saved as .pkl file.
        for file in model_dir.iterdir():
            if file.suffix == ".pkl":
                with file.open("r+b") as fp:
                    return pickle.load(fp)

        # 2nd case: this is reservoirpy v0.2 or v0.1 model
        return reservoirpy.load_compat(model_dir)
    else:
        raise ValueError(f'Unknown model "{name}". '
                         f'Available ESN models: {MODEL_LIST}.')


def _get_config(name: str):
    """Retrieve model configuration.

    Arguments:
    ----------
        name {str} -- Name of the model.

    Raises:
    -------
        FileNotFoundError: No configuration file provided with model.
        ValueError: Bad name or unavailable model.

    Returns:
    --------
        Dict -- Configuration file.
    """
    # retrieve known model configuration
    if name in MODEL_LIST:
        model_dir = os.path.join(MODEL_PATH, name)
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        return config
    # retrieve custom user configuration
    elif os.path.isdir(name):
        try:
            with open(os.path.join(name, "config.json"), "r") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"No 'config.json' configuration "
                                    f"file available in {name}.")
    else:
        raise ValueError(f'Unknown model "{name}". '
                         f'Available ESN models: {MODEL_LIST}.')


def _get_vocab(name: str):
    """Retrieve model configuration.

    Arguments:
    ----------
        name {str} -- Name of the model.

    Raises:
    -------
        FileNotFoundError: No vocab file provided with model.
        ValueError: Bad name or unavailable model.

    Returns:
    --------
        np.ndarray -- Vocab
    """
    # retrieve known model vocab
    if name in MODEL_LIST:
        model_dir = os.path.join(MODEL_PATH, name)
        return np.load(os.path.join(model_dir, "vocab.npy"))

    # retrieve custom user vocab
    elif os.path.isdir(name):
        try:
            return np.load(os.path.join(name, "vocab.npy"))
        except FileNotFoundError:
            raise FileNotFoundError(f"No 'vocab.npy' configuration "
                                    f"file available in {name}.")
    else:
        raise ValueError(f'Unknown model "{name}". '
                         f'Available ESN models: {MODEL_LIST}.')
