<p align="center">
  <img src="assets/logo.png" width="180">
</p>

# Explorica - A Flexible Framework for Exploratory Data Analysis🌱

|         |                                                                                                                                                                                                                                                                                                                            |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI      | ![Tests](https://github.com/laplacedevil/explorica/actions/workflows/ci_tests.yml/badge.svg)![Docs](https://github.com/laplacedevil/explorica/actions/workflows/docs.yml/badge.svg)[![codecov](https://codecov.io/gh/LaplaceDevil/explorica/graph/badge.svg)](https://codecov.io/gh/LaplaceDevil/explorica) |
| Package | ![Python version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue?logo=pypi&logoColor=white)![CondaForge Version](https://img.shields.io/badge/conda--forge-coming%20soon-blue?logo=anaconda&logoColor=green)                                                                                                                                                                          |
| Meta    | [![Documentation](https://img.shields.io/badge/docs-github%20pages-9959d5?logo=githubpages&color=9959d5)](https://laplacedevil.github.io/explorica)[![Documentation](https://img.shields.io/badge/license-MIT-9959d5)](https://github.com/LaplaceDevil/explorica/blob/main/LICENSE.md)                                                                                                                                                                                                                                                               |


**Explorica** is a modular and extensible Python framework for **exploratory data analysis (EDA)**.  
It provides ready-to-use components for **data preprocessing, feature engineering, statistical analysis, visualization and report automation**, allowing analysts and data scientists to focus on insights instead of boilerplate code.

> Designed for data analysts and data scientists who want to streamline their EDA workflow.

---

## Table of Contents
- [Main Features](#main-features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Development Setup](#development-setup)
- [License](#license)

---

## Main Features

- **One-liner Visualizations** - Generate ready-to-use plots for numeric and categorical features with a **single function call**.
- **Beyond Pearson** - Advanced dependency detection using **non-obvious metrics**.
- **Automated EDA Reports** - Run a full EDA pipeline and **generate comprehensive PDF or HTML reports** with **a single script**.

---

## Installation

The source code is currently hosted on GitHub at: https://github.com/LaplaceDevil/explorica.

Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/explorica/).

```bash
# PyPi
pip install explorica
```

Alternatively, for development or to get the latest code:
```bash
# or from github
pip install git+https://github.com/LaplaceDevil/explorica.git
```

---

## Documentation

The official documentation is hosted on [GitHub Pages](https://laplacedevil.github.io/explorica/) - always up-to-date with the latest release.

---

## Development setup

Explorica uses **pylint, flake8 and black** for lint and **pytest** for unit and integration tests. For building documentation, it uses **sphinx, numpydoc, and doctests**.

To set up the development environment:
```bash
git clone https://github.com/LaplaceDevil/explorica
cd explorica

# Basic dev setup
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
pip install -e ".[dev]"

# If you also want to work with documentation
pip install -e ".[dev,docs]"
```

#### Contributing
We welcome contributions of all kinds, including:

- **Bug fixes** - clear and reproducible.
- **New features** - e.g., new visualizations, metrics, or reports.
- **Documentation and examples** - improving docs, tutorials, or demos.
- **Code improvements and tests** - refactoring, optimization, or additional tests.

> Any pull request that follows coding standards and passes tests will be reviewed and merged. We encourage contributors to propose creative ideas and enhancements!

---

## License
[MIT License](LICENSE.md)