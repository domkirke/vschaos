#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

import os
import numpy


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('lnp', parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    sources = ['network_fast.cpp']
    includes = [numpy.get_include()]

    compile_args = ['-O3']

    config.add_extension('network_fast', sources=sources,
                         libraries=libraries,
                         include_dirs=includes,
                         extra_compile_args=compile_args,
                         language='c++')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
