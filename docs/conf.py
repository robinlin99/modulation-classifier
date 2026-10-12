"""Sphinx configuration for the modulation-classifier documentation.

Build with ``make docs`` from the repository root (or ``make html`` here).
"""

import os
import sys
from datetime import date

# Make the package importable for autodoc / sphinx-argparse without installing
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import modulation_classifier  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Modulation Classifier"
author = modulation_classifier.__author__
copyright = f"{date.today().year}, {author}"
release = modulation_classifier.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxarg.ext",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# MyST (Markdown) options
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "substitution"]
myst_heading_anchors = 4

# Autodoc / napoleon
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autoclass_content = "class"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Keep heavy / optional imports from breaking the build on a docs-only host
autodoc_mock_imports = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}
# Don't fail the build if the network is unavailable
intersphinx_timeout = 5

# Silence warnings about unresolved external type names from third-party libs
nitpicky = False
suppress_warnings = ["myst.header"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_show_sourcelink = False
