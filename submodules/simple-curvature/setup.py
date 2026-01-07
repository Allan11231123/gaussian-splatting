# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import subprocess
import pybind11


def pkg_config_cflags(pkgs):
    """用 pkg-config 拿 PCL 的 include flags"""
    try:
        out = subprocess.check_output(
            ["pkg-config", "--cflags"] + pkgs,
            universal_newlines=True
        )
        return out.strip().split()
    except Exception:
        return []


def pkg_config_libs(pkgs):
    """用 pkg-config 拿 PCL 的 linker flags"""
    try:
        out = subprocess.check_output(
            ["pkg-config", "--libs"] + pkgs,
            universal_newlines=True
        )
        return out.strip().split()
    except Exception:
        return []


class BuildExt(build_ext):
    c_opts = {
        "unix": ["-std=c++17", "-O3", "-idirafter", "/usr/include"],
        "msvc": ["/EHsc"],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
        build_ext.build_extensions(self)


# PCL component that you use, commonly seen: common / features / search / kdtree
pcl_pkgs = ["pcl_common-1.12", "pcl_features-1.12", "pcl_search-1.12", "pcl_kdtree-1.12", "eigen3"]
# Note: Based on your PCL installation, the version number (e.g., 1.12) may vary.
# You can check the exact package names using:
#   pkg-config --list-all | grep pcl

cflags = pkg_config_cflags(pcl_pkgs)
libs = pkg_config_libs(pcl_pkgs)

ext_modules = [
    Extension(
        "pcl_curvature",
        ["src/pcl_curvature.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            # "/usr/include",
        ],
        library_dirs=["/usr/lib/x86_64-linux-gnu","/usr/lib"],
        extra_compile_args=cflags,
        extra_link_args=libs,
        language="c++",
    )
]

setup(
    name="pcl_curvature",
    version="0.1.0",
    author="Allan",
    description="PCL-based curvature computation via pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
