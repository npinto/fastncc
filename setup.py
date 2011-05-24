from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules=[ 
    Extension("fastncc_mod", ["fastncc_mod.pyx"],
              include_dirs = [numpy.get_include(),'.'],
              extra_compile_args = ["-O3", "-Wall"],
              ),
]

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
