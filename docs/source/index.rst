Explorica documentation
=======================

|tests| |docs| |license|

.. |tests| image:: https://github.com/fjodordo/explorica/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/fjodordo/explorica/actions/workflows/ci.yml

.. |docs| image:: https://github.com/fjodordo/explorica/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/fjodordo/explorica/actions/workflows/docs.yml

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/fjodordo/explorica/blob/main/LICENSE.md

Explorica is a **modular and extensible** Python framework for exploratory data analysis (EDA).
It provides ready-to-use components for data preprocessing, feature engineering, statistical analysis, and visualization, 
allowing analysts and data scientists to focus on insights instead of boilerplate code.

Key Features
------------

* **Modular Design 🧩**: Use only the components you need for your specific pipeline.
* **Beyond Pearson 🔍**: Advanced dependency detection using Cramér’s V, η², exponential, and other correlation metrics
* **Automation-friendly ⚙️**: Minimize manual work with standardized EDA workflows.

Quick Start
-----------

Installation:

.. code-block:: bash

   pip install git+https://github.com/fjodordo/explorica.git

Basic usage:

.. code-block:: python

   import explorica
   # Your EDA starts here

.. toctree::
   :maxdepth: 2
   :caption: Section Navigation:

   explorica

Indices and tables
==================

* :ref:`modindex`
* :ref:`search`