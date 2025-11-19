# Explorica - A Flexible Framework for Exploratory Data Analysis🌱
![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![Tests](https://github.com/fjodordo/explorica/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)


 **Current version:** 0.1.1 (pre-release)  

**explorica** is a modular and extensible Python framework for **exploratory data analysis (EDA)**.  
It provides ready-to-use components for **data preprocessing, feature engineering, statistical analysis, and visualization**, allowing analysts and data scientists to focus on insights instead of boilerplate code.

> Designed for data analysts and data scientists who want to streamline their EDA workflow.
---
#  Why Explorica?
### Designed for Analyst Productivity
- **One-liner visualizations 📊** - generate ready-to-use plots for numeric and categorical features with a single line of code 

-  **Beyond Pearson 🔍** - advanced dependency detection using Cramér’s V, η², exponential, and other correlation metrics

-  **Smart data quality handling 🧹** - automatically remove or handle NaN values at the column level based on a configurable threshold
---

# Quick Start
```bash
pip install git+https://github.com/fjodordo/explorica.git
```
### Usage: Categorical Correlation Matrix
```python
>>> import pandas as pd
>>> import explorica.interactions as interactions

# Create a small categorical dataset
>>> df = pd.DataFrame({"feat1": ["A", "B", "A", "B"],
...                    "feat2": ["C", "C", "D", "D"],
...                    "feat3": ["A", "A", "B", "B"]})

# Compute Cramér's V correlation matrix
>>> matrix = interactions.corr_matrix_cramer_v(df, bias_correction=False)

>>> print(matrix)
|       | feat1 | feat2 | feat3 |
|-------|-------|-------|-------|
| feat1 | 1.00  | 0.00  | 0.00  |
| feat2 | 0.00  | 1.00  | 1.00  |
| feat3 | 0.00  | 1.00  | 1.00  |

```
### Usage: Search For Highly Correlated Feature Pairs
```python
>>> from seaborn import load_dataset
>>> import explorica.interactions as interactions

# Load Titanic dataset
>>> df = load_dataset("titanic")
 
# NaN removal
>>> df = df.dropna(subset=df.select_dtypes('number').columns)
>>> df = df.dropna(subset=["embark_town"])
>>> df = df.drop("deck", axis=1)

# Separate numeric and categorical features
>>> num_features = df.select_dtypes("number")
>>> cat_features = df.select_dtypes(("object", "category", "bool"))

# Compute highly correlated feature pairs (linear + nonlinear methods)
>>> corr_pairs = interactions.high_corr_pairs(
...     numeric_features=num_features,
...     category_features=cat_features,
...     threshold=0.72,
...     nonlinear_included=True)

# Print results
>>> print(corr_pairs)
| X            | Y          | coef     | method     |
|--------------|------------|----------|------------|
| embark_town  | embarked   | 0.998593 | cramer_v   |
| adult_male   | who        | 0.998593 | cramer_v   |
| who          | adult_male | 0.998593 | cramer_v   |
| embarked     | embark_town| 0.998593 | cramer_v   |
| who          | sex        | 0.933504 | cramer_v   |
| sex          | who        | 0.933504 | cramer_v   |
| adult_male   | sex        | 0.887878 | cramer_v   |
| sex          | adult_male | 0.887878 | cramer_v   |
| fare         | pclass     | 0.745716 | power      |
| fare         | pclass     | 0.743560 | hyperbolic |
| fare         | pclass     | -0.728700| spearman   |
| pclass       | fare       | -0.728700| spearman   |
| fare         | pclass     | 0.725406 | exp        |

``` 
---

# Project Structure

```
explorica/
├── src/explorica/
| |
│ ├── data_quality/
│ │ ├── data_preprocessing.py
│ │ ├── data_quality_handler.py
│ │ ├── feature_engineering.py
| | ├── information_metrics.py
│ │ └── outliers/
| |
│ ├── interactions/
│ │ ├── aggregators.py
│ │ ├── correlation_metrics.py
│ │ ├── correlation_matrices.py
│ │ └── interaction_analyzer.py
| |
│ └── visualizer.py
|
├── tests/
│ ├── unit/
│ └── integration/
|
├── docs/
├── CHANGELOG.md
├── LICENSE.md
├── pyproject.toml
├── README.md
└── requirements.txt
```
- `src/explorica/` — core package, all framework modules are here   
- `docs/` — Sphinx documentation sources  

---

### Functional Modules
- **`interactions`** - top-level functions for analyzing statistical dependencies:
  - **Categorical / Hybrid:** `cramer_v`, `eta_squared`
  - **Numeric:** `corr_index`, `corr_multiple`
  - **Matrix / Vectorized:** `corr_matrix*`, `high_corr_pairs`
- **`data_quality`** - top-level functions for core data handling operations, includes:
  - **outliers** - `describe_distributions`, `remove_outliers`
  - **feature_engineering** - `discretize_continuous`, `freq_encode`
  - **data_preprocessing**  - `get_constant_features`, `get_categorical_features`
  - **information_metrics** - `get_entropy`
- **`visualisations`** — contains **`DataVisualizer`** class
  - generating ready-to-use plots for numeric and categorical data

### Design Principles
- **Modularity** — each component is independent and reusable
- **Extensibility** — easy to add custom logic without changing the core
- **Flat API** — major functions available at top-level modules for easy import
- **Data-Agnostic** — works across domains, not tied to a specific dataset type

---
## Documentation

Documentation is built with Sphinx. To generate locally:
```bash
pip install sphinx sphinx_autodoc_typehints numpydoc
cd docs
make html
open build/html/explorica.html
```

Full documentation will be published on GitHub Pages before the stable release.

---
## Development Setup
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest --cov=explorica

# Code formatting
black src/explorica/

# Code quality checks
black --check src/explorica/
isort --profile black --check-only  src/explorica/
pylint src/explorica/ --max-line-length=88
flake8 src/explorica/ --max-line-length=88
```
---
## Contributing
We welcome:
- Bug reports via [GitHub Issues](https://github.com/fjodordo/explorica/issues)
- Documentation improvements  
- Small fixes

---
## Roadmap
> The roadmap gives a quick overview of completed tasks and future development plans.

- [x] Refactor + n > 80% test coverage for `explorica.data_quality`
- [ ] Refactor + n > 80% test coverage for `explorica.visualizations` 
- [ ] Add new features for `explorica.data_quality`
- [ ] Add `explorica.reports` feature to automate reports
- [ ] Add `explorica.io` feature to load data out of the box
- [ ] PyPI release
- [ ] Prepare demonstration notebooks for release branch

---

## Testing & Continuous Integration
- Unit tests powered by `pytest`, `pytest-cov`
- Code quality enforced via `pylint`, `flake8`, `black`, and `isort`
- Continuous integration via GitHub Actions

---

## Development Status

Explorica is under active development.

---

## Language 💬

The main documentation and code are in English.  
Some commit messages and development notes may include Russian — these will be translated as the project approaches a stable release.

---

## 📜 License

MIT License — see [LICENSE.md](LICENSE.md) for details.