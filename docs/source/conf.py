# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "liesel"
copyright = "2022, Hannes Riebl & Paul Wiemann"
author = "Hannes Riebl, Paul Wiemann"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon", # parse numpy and google style docstrings
    "sphinx.ext.autosummary", # for automatic API doc tables
    # "sphinx.ext.linkcode",
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
pygments_style = "sphinx"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
# html_theme = "pydata_sphinx_theme"
html_title = ""
html_logo = "../../misc/logo/logo-light.png"
html_theme_options = {
    "repository_url": "https://github.com/liesel-devs/liesel",
    "use_repository_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

def linkcode_resolve(domain, info):
    """For the linkcode extension."""
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    
    return f"https://github.com/liesel-devs/liesel/blob/main/{filename}.py"


# maps functions with a class name that is indistinguishable when case is
# ignore to another filename
autosummary_filename_map = {
    "liesel.goose.summary_m.summary": "liesel.goose.summary_m.summary-function",
    "liesel.goose.summary_m.Summary": "liesel.goose.summary_m.summary-class",
}
