# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Assumes docs/ is one level deep


# -- Project information -----------------------------------------------------

project = 'Fault Detection and Classification in Power Systems'
copyright = '2025, Amine Faris'
author = 'Amine Faris'
release = '0.1'  # Optional: Set your project version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
