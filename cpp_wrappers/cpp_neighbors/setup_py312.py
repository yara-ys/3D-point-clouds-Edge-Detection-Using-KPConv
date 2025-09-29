
from setuptools import setup, Extension
import numpy as np
from pathlib import Path
def glob_srcs(root): return [str(p) for p in Path(root).rglob("*.cpp")]
sources = ["wrapper.cpp"] + glob_srcs("neighbors") + glob_srcs("../cpp_utils")
ext = Extension("radius_neighbors", sources=sources,
                include_dirs=[np.get_include(), "neighbors", "../cpp_utils"],
                language="c++", extra_compile_args=["-O3","-std=c++14"])
setup(name="radius_neighbors", version="0.0.0", ext_modules=[ext])
