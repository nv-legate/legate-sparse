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

import os
import sys

# Add project root to sys.path for autodoc to find legate_sparse
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

# -- Project information -----------------------------------------------------

project = "NVIDIA Legate Sparse"
copyright = "2024, NVIDIA"
author = "NVIDIA Corporation"

# TODO: add version switcher similar to cuPyNumeric
version = release = "26.02"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
    "nbsphinx",
    "legate._sphinxext.settings",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- RST Epilog for centralized link definitions -----------------------------
# No need to repeat these in individual RST files

rst_epilog = """
.. _Legate: https://github.com/nv-legate/legate
.. _cuPyNumeric: https://github.com/nv-legate/cupynumeric
.. _CuPyNumeric: https://github.com/nv-legate/cupynumeric
.. _NumPy: https://numpy.org/doc/stable/index.html
.. _scipy: https://docs.scipy.org/doc/scipy/index.html
.. _scipy.sparse: https://docs.scipy.org/doc/scipy/tutorial/sparse.html
.. _scipy.spatial: https://docs.scipy.org/doc/scipy/tutorial/spatial.html
.. _Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
.. _examples: https://github.com/nv-legate/legate-sparse/tree/HEAD/examples
.. _Running Legate Programs: https://docs.nvidia.com/legate/latest/usage.html#running-legate-programs
.. _Resource Allocation: https://docs.nvidia.com/legate/latest/usage.html#resource-allocation
.. _Usage: https://docs.nvidia.com/legate/latest/usage.html#usage
.. _common.py: https://github.com/nv-legate/legate-sparse.internal/tree/main/examples/common.py
.. _Building Legate From Source: https://docs.nvidia.com/legate/latest/BUILD.html#building-from-source
.. _Building CuPyNumeric From Source: https://docs.nvidia.com/cupynumeric/latest/developer/building.html#building-from-source
"""

# -- Options for HTML output -------------------------------------------------

html_context = {
    # "default_mode": "light",
    "AUTHOR": author,
    "DESCRIPTION": "Legate Sparse documentation site.",
}

html_static_path = ["_static"]

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "announcement": (
        "This project has reached end of life and is no longer maintained or "
        "supported. The final public release is 26.02.00. This documentation "
        "is retained for historical reference."
    )
}

templates_path = ["_templates"]

# -- Options for extensions --------------------------------------------------

autosummary_generate = True

copybutton_prompt_text = ">>> "

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

napoleon_custom_sections = [("Availability", "returns_style")]

nbsphinx_execute = "never"

pygments_style = "sphinx"


def setup(app):
    pass
