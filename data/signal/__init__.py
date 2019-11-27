# -*-coding:utf-8 -*-
 
"""
    The ``datasets`` module
    ========================
 
    This package contains all datasets classes
 
    :Example:
 
    >>> from data.sets import DatasetAudio
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
__author__  = "esling@ircam.fr"
__date__    = ""
__all__     = ["modulation", "strf", "transforms"]
# -*- coding: utf-8 -*-
from . import nsgt3 as nsgt
from .transforms import transformHandler, inverseTransform


