#!/usr/bin/env python
#coding:utf-8
# Author:  mozman
# to build c-extension:
# setup.py build_ext --inplace

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



ext_modules = \
  [
    Extension("llmc.sparse", ["llmc/sparse.pyx", "llmc/hash-table.c"])
  ]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'llmc',
    version='0.1.0',
    description='Efficient Low-Level Implementation of MCMC Algorithm',
    author='csy',
    author_email='chenshouyuan@gmail.com',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    packages=['llmc'],
    requires=['cython']
)
