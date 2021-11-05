"""
Canary Decoder
##############

Automatic annotation of acoustic signals from songbirds.
"""

from . import utils

from .extraction import extract
from .decoding.decoder import Decoder, load
from .extraction.processor import Processor