# 📊 explorica — A Flexible Framework for Exploratory Data Analysis

 **Current version:** 0.0.4 (pre-release)  

**explorica** is a modular and extensible Python framework for **exploratory data analysis (EDA)**.  
It provides ready-to-use components for **data preprocessing, feature engineering, statistical analysis, and visualization**, allowing analysts and data scientists to focus on insights instead of boilerplate code.

---

## Roadmap

- [ ] Integrate Continuous Integration (CI) for automated testing and linting
- [ ] Implement comprehensive unit and integration tests
- [ ] Create DataQualityHandler module combining preprocessing, outlier handling, and feature engineering
- [ ] Prepare demonstration notebooks for release branch

---

## 📂 Project Structure

```
explorica/
├── src/
│ ├── explorica/
│ │ ├── config/
│ │ │ └── messages.json
│ │ ├── __init__.py
│ │ ├── _utils.py
│ │ ├── data_preprocessor.py
│ │ ├── feature_engineer.py
│ │ ├── interaction_analyzer/
│ │ │ ├── __init__.py
│ │ │ ├── aggregators.py
│ │ │ ├── correlation_metrics.py
│ │ │ └── interaction_analyzer.py
│ │ ├── outlier_handler.py
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

---

## 🛠 Components

### Core Classes
- **`DataPreprocessor`** — handling missing values, type casting, basic transformations
- **`FeatureEngineer`** — creating, encoding, and transforming features
- **`InteractionAnalyzer`** — detecting and quantifying relationships between variables
- **`OutlierHandler`** — identifying and processing outliers
- **`DataVisualizer`** — generating ready-to-use plots for numeric and categorical data

### Design Principles
- **Modularity** — each component is independent and reusable
- **Extensibility** — easy to add custom logic without changing the core
- **Data-Agnostic** — works across domains, not tied to a specific dataset type


## 🚧 Development Status

This project is under active development and will grow iteratively.  
It’s part of my journey toward building better data understanding, research tools, and ML-ready pipelines.

---

## 💬 Language

The main documentation and code are in English.  
Some commit messages and development notes may include Russian — these will be translated as the project approaches a stable release.

---

## 📜 License

MIT License — see [LICENSE.md](LICENSE.md) for details.