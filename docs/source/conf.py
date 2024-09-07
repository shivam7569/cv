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


# a = os.path.abspath("backbones")

sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath("../../backbones"))
# sys.path.insert(0, os.path.abspath("backbones/LeNet"))
# sys.path.insert(0, os.path.abspath("backbones/AlexNet"))
# sys.path.insert(0, os.path.abspath("backbones/VGG16"))
# sys.path.insert(0, os.path.abspath("../../attention"))
# sys.path.insert(0, os.path.abspath("../../src"))

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

templates_path = ['_templates']
exclude_patterns = []

master_doc = 'index'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_material"

html_static_path = ['_static']
html_show_sphinx = False
