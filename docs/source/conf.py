import sys, os

sys.path.insert(0, os.path.abspath('../src/explorica'))

project = 'Explorica'
copyright = '2026, fjodordo'
author = 'fjodordo'
release = '0.11.4'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'numpydoc',
]

autodoc_default_options = {
    'members': True,
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

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 10,
    'includehidden': True,
    'titles_only': False,
}


suppress_warnings = ["ref.class", "ref.func"]

templates_path = ['_templates']
exclude_patterns = []


html_static_path = ['_static']
