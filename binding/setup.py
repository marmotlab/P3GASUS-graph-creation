from setuptools import setup, Extension
import pybind11
import sys
import os

pybind_include = pybind11.get_include()

discrete_ext = Extension(
    "p3gasus_discrete_cpp",
    sources=["discrete_bindings.cpp"],
    include_dirs=[pybind_include, "."],
    language="c++",
    extra_compile_args=[
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
    ],
)

continuous_ext = Extension(
    "p3gasus_continuous_cpp",
    sources=["continuous_bindings.cpp"],
    include_dirs=[pybind_include, "."],
    language="c++",
    extra_compile_args=[
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
    ],
)

setup(
    name="p3gasus",
    version="1.0.0",
    description="C++ ADG backends for discrete and continuous P3GASUS graph construction",
    ext_modules=[discrete_ext, continuous_ext],
    python_requires=">=3.8",
)
