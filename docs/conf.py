project = "torchscience"
copyright = "2024-2026, Allen Goodman"
author = "Allen Goodman"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "alphabetical"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
