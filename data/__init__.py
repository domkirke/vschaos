# -*-coding:utf-8 -*-
 
"""
    The ``datasets`` module
    ========================
 
    This package contains all datasets classes
 
    :Example:
 
    >>> from data import DatasetAudio
    >>> DatasetAudio()
 
    Subpackages available
    ---------------------

        * Generic
        * Audio
        * Midi
        * References
        * Time Series
        * Pytorch
        * Tensorflow
 
    Comments and issues
    ------------------------
        
        None for the moment
 
    Contributors
    ------------------------
        
    * Philippe Esling       (esling@ircam.fr)
 
"""
 
# info
__version__ = "1.0"
__author__  = "esling@ircam.fr, chemla@ircam.fr"
__date__    = ""
__all__     = ["Dataset", "DatasetAudio", "metadata", "pp", "toys", "signal"]

# import sub modules
import pdb
#pdb.set_trace()
from .data_utils import *
from .data_generic import Dataset, dataset_from_torch
from .data_audio import DatasetAudio, OfflineDatasetAudio
from . import data_metadata as metadata
from .data_preprocessing import Normalize, Magnitude
from . import toys
from . import signal
