# -*-coding:utf-8 -*-
 
"""
    The ``utils`` module
    ========================
 
    This package contains all utility and side functions and classes
 
    Examples
    --------
 
    Subpackages available
    ---------------------
 
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
__all__     = []
 
# import sub modules
from .onehot import oneHot, fromOneHot
from .cage_deform import SerieDeformation
from .misc import *
from .utils_modules import *
from .gather_distrib import *
from .scatter import *
from . import oscServer
