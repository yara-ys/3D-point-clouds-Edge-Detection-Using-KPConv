
from setuptools import setup, Extension
import numpy as np
from pathlib import Path
def glob_srcs(root): return [str(p) for p in Path(root).rglob("*.cpp")]
sources = ["wrapper.cpp"] + glob_srcs("grid_subsampling") + glob_srcs("../cpp_utils")
ext = Extension("grid_subsampling", sources=sources,
                include_dirs=[np.get_include(), "grid_subsampling", "../cpp_utils"],
                language="c++", extra_compile_args=["-O3","-std=c++14"])
setup(name="grid_subsampling", version="0.0.0", ext_modules=[ext])
