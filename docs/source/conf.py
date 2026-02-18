import sys, os

sys.path.insert(0, os.path.abspath('../src/explorica'))

project = 'Explorica'
copyright = '2026, fjodordo'
author = 'fjodordo'
release = '0.11.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
]

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 10,
    'includehidden': True,
    'titles_only': False,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_preprocess_types = True

templates_path = ['_templates']
exclude_patterns = []


html_static_path = ['_static']
