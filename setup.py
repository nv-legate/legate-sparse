#!/usr/bin/env python3

# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from setuptools import find_packages
from skbuild import setup

# TODO: build yields "cant find legate module"....
"""
import legate.install_info as lg_install_info
import os
from pathlib import Path

legate_dir = Path(lg_install_info.libpath).parent.as_posix()

cmake_flags = [
    f"-Dlegate_ROOT:STRING={legate_dir}",
]

env_cmake_args = os.environ.get("CMAKE_ARGS")
if env_cmake_args is not None:
    cmake_flags.append(env_cmake_args)
os.environ["CMAKE_ARGS"] = " ".join(cmake_flags)
"""

setup(
    name="legate-sparse",
    version="24.11",
    description="An Aspiring Drop-In Replacement for SciPy Sparse module at Scale",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(
        where=".",
        include=["legate_sparse*"],
    ),
    include_package_data=True,
    zip_safe=False,
)
