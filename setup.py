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

libs = {"gsl": (include_gsl_dir, lib_gsl_dir)}
srcs = ["llmc/lib/hash-table.c"]
exts = [("llmc.spmatrix", "llmc/spmatrix.pyx", []),
        ("llmc.py_spmatrix", "llmc/py_spmatrix.pyx", []),
        ("llmc.model.topicmodel", "llmc/model/topicmodel.pyx", ["gsl"]),
        ("llmc.model.mixture", "llmc/model/mixture.pyx", [])]

ext_modules = []
for mod, pyx, lib in exts:
  include_dirs = [libs[l][0] for l in lib]
  library_dirs = [libs[l][1] for l in lib]
  ext = Extension(mod, [pyx]+srcs,
                  include_dirs=include_dirs,
                  library_dirs=library_dirs,
                  libraries = lib)
  ext_modules.append(ext)

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
