'''
python setup.py build_ext -i to compile
'''

import platform
from distutils.core import setup, Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

compile_extra_args = []
link_extra_args = []

if platform.system()=='Windows':
    compile_extra_args = ['/std:c++latest', '/EHsc']
elif platform.system()=='Darwin':
    compile_extra_args = ['-std=c++11', '-stdlib=libc++',
                          '-mmacosx-version-min=10.9']
    link_extra_args = ['-stdlib=libc++', '-mmacosx-version-min=10.9']


setup(
    name='mesh_core_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("mesh_core_cython",
                           sources=["mesh_core_cython.pyx", "mesh_core.cpp"],
                           language='c++',
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=compile_extra_args,
                           extra_link_args=link_extra_args)],
)
