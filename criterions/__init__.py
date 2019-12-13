
# -*-coding:utf-8 -*-
 
"""
    The ``criterions`` module
    ========================
 
    This package contains different criterions and criterion components for VAE training
 

    Comments and issues
    ------------------------
        
        None for the moment
 
    Contributors
    ------------------------
        
    * Axel Chemla--Romeu-Santos (chemla@ircam.fr)
 
"""
 
# info
__version__ = "0.1.0"
__author__  = "chemla@ircam.fr"
__date__    = "11/03/19"

# import sub modules
from .. import utils
from .criterion_criterion import *
from .criterion_logdensities import *
from .criterion_divergence import *
from .criterion_functional import *
from .criterion_elbo import *
from .criterion_scan import *
from .criterion_misc import *
from .criterion_adversarial import Adversarial

