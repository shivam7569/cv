# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = 'cv'
copyright = '2024, Shivam Chaudhary'
author = 'Shivam Chaudhary'
release = '0.0.1'

sys.path.insert(0, os.path.abspath("../../cv"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon'
]

autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
html_show_sphinx = False

def skip_member(app, what, name, obj, skip, options):
    if what == "class" and name in ["BoT_ViTParams", "ResidualGroup", "ResidualBlock", "ConvBlock"]:
        return True
    # if what == "method" and name in ["method_to_exclude1", "method_to_exclude2"]:
        # return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)