# Explorica — A Flexible Framework for Exploratory Data Analysis🌱

 **Current version:** 0.1.0 (pre-release)  

**explorica** is a modular and extensible Python framework for **exploratory data analysis (EDA)**.  
It provides ready-to-use components for **data preprocessing, feature engineering, statistical analysis, and visualization**, allowing analysts and data scientists to focus on insights instead of boilerplate code.

> Designed for data analysts and data scientists who want to streamline their EDA workflow.
---

# Quick Start
### Clone the repository and install dependencies:
```bash
git clone https://github.com/LaplaceDevil/explorica.git
cd explorica
pip install -r requirements.txt
```
### Usage: Categorical Correlation Matrix
```python
>>> import pandas as pd
>>> from explorica.interactions import corr_matrix_cramer_v

# Create a small categorical dataset
>>> df = pd.DataFrame({"feat1": ["A", "B", "A", "B"],
...                    "feat2": ["C", "C", "D", "D"],
...                    "feat3": ["A", "A", "B", "B"]})

# Compute Cramér's V correlation matrix
>>> matrix = corr_matrix_cramer_v(df, bias_correction=False)

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
>>> from explorica.interactions import high_corr_pairs

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
>>> corr_pairs = high_corr_pairs(
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

# Project Structure 📂

```
explorica/
├── src/
│ ├── explorica/
│ │ ├── config/
│ │ │ └── messages.json
│ │ ├── __init__.py
│ │ ├── _utils.py
│ │ ├── data_quality/
│ │ | ├── __init__.py
│ │ │ ├── data_preprocessor.py
│ │ │ ├── data_quality_handler.py
│ │ │ ├── feature_engineer.py
│ │ │ └── outlier_handler.py
│ │ ├── interactions/
│ │ │ ├── __init__.py
│ │ │ ├── aggregators.py
│ │ │ ├── correlation_metrics.py
│ │ │ ├── correlation_matrices.py
│ │ │ └── interaction_analyzer.py
│ │ └── visualizer.py
├── tests/
│ ├── unit/
│ │ └── test_interaction_analyzer.py
│ └── __init__.py
├── docs/
| ├── source/
| | ├── conf.py
| | ├── index.rst
| | └── ...
| ├── make.bat
| └── Makefile
├── .github/workflows/
| └── ci.yml
├── .gitignore
├── CHANGELOG.md
├── LICENSE.md
├── pyproject.toml
├── pytest.ini
├── README.md
└── requirements.txt
```
- `src/explorica/` — core package, all framework modules are here  
- `config/` — static configuration (e.g. `messages.json`)  
- `docs/` — Sphinx documentation sources  
- `.github/workflows/` — CI/CD pipeline configs  

---

# Components

### Core Classes
- **`DataVisualizer`** — generating ready-to-use plots for numeric and categorical data

### Functional Modules
- **`interactions`** — top-level functions for analyzing statistical dependencies:
  - **Categorical / Hybrid:** `cramer_v`, `eta_squared`
  - **Numeric:** `corr_index`, `corr_multiple`
  - **Matrix / Vectorized:** `corr_matrix*`, `high_corr_pairs`
- **`data_quality`** — unified facade for core data handling operations, includes:
  - `DataPreprocessor` — handling missing values, managing categories, detecting constant features
  - `FeatureEngineer` — creating, encoding, and transforming features
  - `OutlierHandler` — detecting and processing outliers, describing distributions

### Design Principles
- **Modularity** — each component is independent and reusable
- **Extensibility** — easy to add custom logic without changing the core
- **Flat API** — major functions available at top-level modules for easy import
- **Data-Agnostic** — works across domains, not tied to a specific dataset type

---

# Highlights 🌠

- **One-liner visualizations 📊** — generate ready-to-use plots for numeric and categorical features with a single line of code 

-  **Beyond Pearson 🔍** — advanced dependency detection using Cramér’s V, η², exponential, and other correlation metrics

-  **Smart data quality handling 🧹** — automatically remove or handle NaN values at the column level based on a configurable threshold
---
## Roadmap
> The roadmap gives a quick overview of completed tasks and future development plans.

- [x] Integrate Continuous Integration (CI) for automated testing and linting
- [x] Implement a basic set of unit tests for explorica.interactions
- [ ] Cover 80%+ of Explorica with unit tests
- [ ] Create DataQualityHandler module combining preprocessing, outlier handling, and feature engineering
- [ ] Prepare demonstration notebooks for release branch

---

## Testing & Continuous Integration
- Unit tests powered by `pytest`
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