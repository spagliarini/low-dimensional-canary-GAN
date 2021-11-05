"""
`canarydecoder.Decoder`
-----------------------

Audio annotation pipeline for songbirds.
"""

import os
import time
import inspect
from typing import Union, Optional, Dict, Sequence

import numpy as np
import reservoirpy

from .annotation import Annotation
from ._models import _get_model, _get_config, _get_vocab
from ..extraction.processor import Processor


class Decoder(object):

    @staticmethod
    def _processor_from_config(config):
        # fetch all non default parameters (they are required in the configuration dict)
        required_args = [k for k, v in inspect.signature(Processor.__init__).parameters.items() 
                         if v.default is not inspect.Parameter.empty]
        
        for arg in required_args:
            if arg not in config.keys():
                raise ValueError(f"Missing parameter {arg} in processor config file.")
        
        return Processor(**config)
            
    
    def __init__(self, esn: reservoirpy.ESN, inputs: Dict, vocab: np.ndarray, 
                 continuous: bool, processor_config: Dict):
        """Decoder pipeline.

        Arguments:
        ----------
            esn {reservoirpy.ESN} -- ESN model used to run predictions.
            
            inputs {Dict} -- Specifications of expected model inputs.
            
            vocab {Dict} -- Vocabulary for data decoding.
            
            continuous {bool} -- If `True`, will will run the decoder over all concatenated inputs.
            
            processor_config {Dict} -- Configuration of `Processor` object.
            
        Example:
        --------
        ::
            from canarydecoder import load
            decoder = load("canary16")
            annotations = decoder("./samples/of/songs")
        """
        self._esn = esn
        self._processor = Decoder._processor_from_config(processor_config)
        self.inputs = inputs
        self.continuous = continuous
        self.vocab = vocab
    
    def __repr__(self):
        s = f"Decoder : {str(self._esn)}\n{str(self._processor)}\n"
        s += f"Inputs: {[k for k in self.inputs.keys()]} Output size: {self._esn.dim_out}"
        return s
    
    
    def __call__(self, waves, verbose=False, workers=-1, backend="threading"):
        # Main method to use the decoder
        # First call the processor on the specifiyed files or arrays, then
        # run the models on the extracted features.
        if verbose: 
            print("Extracting features")
            tic = time.time()
            
        all_features, all_waves, all_files = self._processor(waves, workers=workers, backend=backend, **self.inputs)
        
        all_features = [np.hstack(feat) for feat in all_features]
        if self.continuous:
            all_features = [np.vstack(all_features)]
        
        if verbose: 
            toc = time.time()
            print(f"Done ({toc - tic:.3f}s)")
        
        raw_outputs, states = self._esn.run(all_features, verbose=verbose, workers=workers, backend=backend)
        
        mean_outputs = np.array([o.mean(axis=0) for o in raw_outputs])
        
        annotations = [Annotation(w, i, f, o, self.vocab, self._processor._hop_length) 
                       for w, i, f, o in zip(all_waves, all_files, all_features, raw_outputs)] 
        
        return annotations
        

def load(model: str) -> Decoder:
    """Load an existing pre-trained decoder.

    Arguments:
    ----------
        model {str} -- Path to the decoder model directory.
        
    Returns:
    --------
        Decoder -- Decoder object with loaded Processor and model.
        
    Notes:
    ------
    
    Available models:
    
    * `canary16`: 16 syllables canary decoder with ESN.
    * `canary16-deltas`: 16 syllables canary decoder with ESN (only signal derivatives).
    * `canarygan-3`: 16 syllables + 3 GAN generations from 3 different epochs.
    * `canarygan-3-d`: 16 syllables + 3 GAN generations from 3 different epochs (only signal derivatives).
    """
        
    esn = _get_model(model)
    config = _get_config(model)
    vocab = _get_vocab(model)
    
    processor_config = config["preprocessing"]
    continuous = config["inputs"]["continuous"]
    inputs = {k: v for k, v in config["inputs"].items() if k != "continuous"}
    
    return Decoder(esn, inputs, vocab, continuous, processor_config)