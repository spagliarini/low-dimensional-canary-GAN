"""
canarydecoder.extract
---------------------

Audio handling and features extraction module.
"""

import os
from typing import Sequence

import joblib
import librosa as lbr
import numpy as np

from ..utils import parallel_on_demand


def load_wave(x_path: str, sr: int=None) -> np.ndarray:
    """Load a single .wav file.

    Arguments:
    ----------
        x_path {str} -- Path to the .wav file.

    Keyword Arguments:
    ------------------
        sr {int} -- Sampling rate of the audio signal. (default: {None})

    Returns:
    --------
        np.ndarray -- Audio signal data.
    """
    x, _ = lbr.load(x_path, sr=sr)
    return x

    
def extract_features(wave: np.ndarray, 
                     sr: int=None, hop_length: int=512, 
                     n_fft: int=1024, trim_below_db: int=20, 
                     padding: str='wrap', mfcc: bool=True, 
                     delta1: bool=True, delta2: bool=True, 
                     lifter: int = 0) -> Sequence[np.ndarray]:
    """Extract MFCC, and derivatives of MFCC, from an audio signal.

    Arguments:
    ----------
        wave {np.ndarray} -- Audio signal.

    Keyword Arguments:
    ------------------
        sr {int} -- Sampling rate of audio signal. (default: {None})
        hop_length {int} -- Number of samples between succesive frames. (default: {512})
        n_fft {int} -- Number of samples within a frame for spectrum computation. (default: {1024})
        trim_below_db {int} -- Log power threshold below which signal is cut. (default: {20})
        padding {str} -- Padding mode for derivatives computation. (default: {'wrap'})
        mfcc {bool} -- If True, compute MFCC. (default: {True})
        delta1 {bool} -- If True, compute 1st MFCC derivative. (default: {True})
        delta2 {bool} -- If True, compute 2nd MFCC derivative. (default: {True})

    Returns:
    --------
        Sequence[np.ndarray] -- Extracted features.
        
    Notes:
    ------
    
    See `librosa <https://librosa.github.io/librosa/index.html>`_ documentation for more informations.
    """

    w = wave
    if trim_below_db > 0:
        w, _ = lbr.effects.trim(wave, top_db=trim_below_db)

    mfcc_sig = lbr.feature.mfcc(w, sr, hop_length=hop_length, n_fft=n_fft, lifter=lifter)
    
    features = []
    if mfcc:
        features.append(mfcc_sig.T)
    if delta1:
        features.append(lbr.feature.delta(mfcc_sig, mode=padding).T) 
    if delta2:
        features.append(lbr.feature.delta(mfcc_sig, order=2, mode=padding).T)
    
    return features


def load_all_waves(directory: str, sr: int=None, workers: 
                   int=None, backend: str="threading") -> Sequence[np.ndarray]:
    """Load all waves stored in a directory.

    Arguments:
    ----------
        directory {str} -- Directory containing the .wav files.

    Keyword Arguments:
    ------------------
        sr {int} -- Sampling rate. (default: {None})
        workers {int} -- number of threads/processes to use for parallel execution. (default: {None})
        backend {str} -- joblib backend. (default: {"threading"})

    Raises:
    -------
        FileNotFoundError: The directory does not contains any .wav file to load.

    Returns:
    --------
        Sequence[np.ndarray] -- Arrays of audio signals.
    """
    
    all_waves_files = [os.path.join(directory, f) 
                       for f in os.listdir(directory) 
                       if os.path.splitext(f)[1] == '.wav']
    
    if len(all_waves_files) < 1:
        raise FileNotFoundError(f"No .wav files found in {directory}.")
    
    loop = joblib.Parallel(n_jobs=workers, backend=backend)
    delayed_wave = joblib.delayed(load_wave)
    
    all_waves = loop(delayed_wave(f, sr=sr)
                     for f in all_waves_files)

    return all_waves, all_waves_files