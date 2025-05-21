Installation Guide
==================

This document provides instructions to set up the environment required to run the **TimeSeriesProject3.0**.

Prerequisites
-------------

Ensure you have Python 3.8 or higher installed. It is recommended to use a virtual environment.

You can create and activate a virtual environment as follows:

.. code-block:: bash

   python -m venv venv
   source venv\Scripts\activate 

Install Required Packages
-------------------------

Install the required Python packages using `pip`:

.. code-block:: bash

   pip install -r requirements.txt

If a `requirements.txt` is not available, manually install the dependencies listed below.

Dependencies
------------

The following Python libraries are used in this project:

.. code-block:: bash

   numpy
   pandas
   matplotlib
   seaborn
   scikit-learn
   scipy
   statsmodels
   pmdarima
   tensorflow
   keras
   joblib

Optional (for notebook execution):

.. code-block:: bash

   jupyter
   ipywidgets

You can install them using pip:

.. code-block:: bash

   pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels pmdarima tensorflow keras joblib jupyter ipywidgets

Run the Project
---------------

To run the notebook:

.. code-block:: bash

   jupyter notebook TimeSeriesProject3.0.ipynb

This will open the notebook in your default web browser. Follow the cells in sequence to reproduce the analysis.

Troubleshooting
---------------

- Ensure all dependencies are installed before running the notebook.
- For TensorFlow GPU support, ensure CUDA and cuDNN are properly configured.
- Use `pip list` to verify installed package versions.


