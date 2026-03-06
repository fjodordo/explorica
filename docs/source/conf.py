import sys
from pathlib import Path


# sys.path.insert(0, os.path.abspath('../src/explorica'))

sys.path.insert(0, str(Path("../../src").resolve()))

import explorica

project = "Explorica"
copyright = "2026, LaplaceDevil"
author = "LaplaceDevil"
release = explorica.__version__
version = explorica.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "numpydoc",
]

with open("../_links.rst") as f:
    rst_prolog = f.read()

autodoc_default_options = {
    "members": True,
}

numpydoc_validation_checks = {
    "all",
    "SA01", # See also section is missing
    "YD01", # Yields section is missing
    "PR01", # Parameters section not documented (disabled because a kwargs issue)
    "PR02", # Unknown parameters documented (disabled because a kwargs issue)
}

numpydoc_validation_exclude = {
    r"explorica\._utils.*",
}

html_theme = "sphinx_rtd_theme"
html_logo = "../../assets/logo_stroke.svg"
html_favicon = "../../assets/logo_small.png"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 10,
    "includehidden": True,
    "titles_only": False,
}


suppress_warnings = ["ref.class", "ref.func"]

templates_path = ["_templates"]
exclude_patterns = []


# html_static_path = ["_static"]
