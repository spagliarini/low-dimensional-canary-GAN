"""
`canarydecoder.Processor`
-------------------------

Processor object for data loading and feature extraction.
"""

import os

import joblib
import librosa as lbr
import numpy as np

from .extract import extract_features, load_all_waves, load_wave


class Processor(object):


    def __init__(self, sampling_rate: int, n_fft: int, 
                 hop_length: int, padding: str, trim_below_db: int,
                 lifter: int = 0):
        """Processor object for data loading and feature extraction.

        Arguments:
        ----------
            sampling_rate {int} -- Sampling rate to apply.
            n_fft {int} -- Frame size for FFT computation.
            hop_length {int} -- Number of samples between each frame.
            padding {str} -- Padding mode for derivatives.
            trim_below_db {int} -- Log power threshold below which the signal is cut.
        """
        
        self._sr = sampling_rate
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._padding = padding
        self._top_db = trim_below_db   
        self._lifter = lifter
            
    def __repr__(self):
        return f"Processor({[str(k)+'='+str(v) for k, v in self.__dict__.items()]})"
            
            
    def __call__(self, waves, mfcc=True, delta1=True, delta2=True, 
                 workers=-1, backend="threading"):
        # Load waves, and extract features from them.
        if type(waves) is str:
            if os.path.isdir(waves):
                all_waves, all_files = load_all_waves(waves, sr=self._sr)
            elif os.path.isfile(waves):
                all_files = [waves]
                all_waves = [load_wave(waves, sr=self._sr)]
            else:
                raise FileNotFoundError(f"File or directory {waves} not found.")
        else:
            if type(waves) is np.ndarray:
                all_files = [0]
                all_waves = [waves]
            else:
                all_files = [*range(len(waves))]
                all_waves = waves
        
        loop = joblib.Parallel(n_jobs=workers, backend=backend)
        delayed_features = joblib.delayed(extract_features)
        
        all_features = loop(delayed_features(w, sr=self._sr, hop_length=self._hop_length, 
                                             n_fft=self._n_fft, padding=self._padding, 
                                             trim_below_db=self._top_db, lifter=self._lifter,
                                             mfcc=mfcc, delta1=delta1, delta2=delta2)
                            for w in all_waves)
        
        return all_features, all_waves, all_files