#!/usr/bin/env python
#coding:utf-8
# Author:  mozman
# to build c-extension:
# setup.py build_ext --inplace

from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext

from os.path import splitext, basename, normpath, join as pjoin
import sys, os

if sys.platform in ["linux2", "darwin"]:
  include_gsl_dir = "/usr/local/include/"
  lib_gsl_dir = "/usr/local/lib/"
elif sys.platform == "win32":
  include_gsl_dir = r"D:\code\gsl\gsl-1.11\include"
  lib_gsl_dir = r"D:\code\gsl\gsl-1.11\lib-static"


builtin_c_lib = normpath("llmc/lib")

class CleanCommand(Command):
  """Custom distutils command to clean the .so and .pyc files."""
  user_options = [ ]

  def initialize_options(self):
    self._clean_me = ['tags']
    for root, dirs, files in os.walk('.'):
       for f in files:
          if f.endswith('.pyc') or f.endswith('.so') or f.endswith('.pyd'):
            self._clean_me.append(pjoin(root, f))
          elif f.endswith('.c') and normpath(root) != builtin_c_lib:
            self._clean_me.append(pjoin(root, f))

  def finalize_options(self):
    pass

  def run(self):
    import shutil
    try:
      shutil.rmtree('build')
    except:
      pass

    for clean_me in self._clean_me:
      try:
        os.unlink(clean_me)
      except:
        pass


ext_modules = \
  [
    Extension("llmc.spmatrix",
              ["llmc/spmatrix.pyx",
               "llmc/lib/hash-table.c"]),
    Extension("llmc.builtin.topicmodel",
              ["llmc/builtin/topicmodel.pyx",
               "llmc/lib/hash-table.c"],
              include_dirs = [include_gsl_dir],
              library_dirs = [lib_gsl_dir],
              libraries = ["gsl"] ),
    Extension("llmc.builtin.mixture",
              ["llmc/builtin/mixture.pyx",
               "llmc/lib/hash-table.c"])
  ]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'llmc',
    version='0.1.0',
    description='Efficient Low-Level Implementation of MCMC Algorithm',
    author='csy',
    author_email='chenshouyuan@gmail.com',
    cmdclass = {'build_ext': build_ext, 'clean':CleanCommand},
    ext_modules = ext_modules,
    packages=['llmc'],
    requires=['cython']
)
