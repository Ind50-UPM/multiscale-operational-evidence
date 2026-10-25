Getting started
===============

Requirements
------------

* Python 3.10 or newer (the release was validated with Python 3.12, NumPy 2.2,
  pandas 3.0, pyarrow 17 and scikit-learn 1.5).
* `Git LFS <https://git-lfs.com>`_ to retrieve the versioned result tables
  (``*.csv`` and ``*.parquet`` are stored with LFS).
* Graphviz (only to build the UML diagrams of this documentation).

Clone and install
-----------------

.. code-block:: bash

   git lfs install
   git clone https://github.com/Ind50-UPM/multiscale-operational-evidence.git
   cd multiscale-operational-evidence
   python -m venv .venv && source .venv/bin/activate
   pip install -e src/Model_A -e src/Model_B -e src/Model_C -e "src/Model_D[experiments]"

Each package can also be installed with Poetry from its own folder
(``poetry install``).

Data
----

The raw telemetry (``data/*.parquet``, about 1.8 GB) is distributed with Git
LFS; ``git lfs pull`` downloads it. See ``data/README.md`` for the files and
their schemas. The derived artifacts needed to reproduce every table of the
manuscript (incident registry, ablation, multiscale and extension outputs) are
versioned under ``src/predictions/Model_D`` and ``reports/``.

Run the tests
-------------

.. code-block:: bash

   for pkg in Model_A Model_B Model_C Model_D; do (cd src/$pkg && python -m pytest -q tests); done

Reproduce the manuscript experiments
------------------------------------

.. code-block:: bash

   export PYTHONPATH=src:src/Model_D
   python src/run_model_d_ablation.py            # matched M0–M3 ablation (24 h)
   python src/run_model_d_multiscale.py          # 15 min / 1 h / 6 h / 24 h sensitivity
   python src/run_model_d_extensions.py --stage all
   python src/export_extension_figure_data.py    # pgfplots tables used by the paper

The full execution order, inputs and outputs are listed in
:doc:`reproducibility`; the mapping between manuscript sections and code is in
:doc:`paper_map`.

Build this documentation
------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
