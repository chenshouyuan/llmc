#!/usr/bin/env python
#coding:utf-8
# Author:  mozman
# to build c-extension:
# setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import sys, os

if sys.platform == "linux2" :
  include_gsl_dir = "/usr/local/include/"
  lib_gsl_dir = "/usr/local/lib/"
elif sys.platform == "win32":
  include_gsl_dir = r"D:\code\gsl\gsl-1.11\include"
  lib_gsl_dir = r"D:\code\gsl\gsl-1.11\lib-static"

ext_modules = \
  [
    Extension("llmc.sparse",
              ["llmc/sparse.pyx",
               "llmc/hash-table.c"],
              include_dirs = [include_gsl_dir],
              library_dirs = [lib_gsl_dir],
              libraries = ["gsl"] )
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
