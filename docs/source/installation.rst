.. _installation:

===================
Installation Guide
===================

This document provides complete instructions to set up the **Power Grid Anomaly Detection System** on your local machine or server.

System Requirements
-------------------

Minimum:
- Python 3.8+
- 8GB RAM
- 10GB disk space
- Windows 10+/macOS 10.15+/Linux (Ubuntu 20.04+ recommended)

Recommended for optimal performance:
- Python 3.10
- 16GB+ RAM
- NVIDIA GPU with CUDA support (for TensorFlow acceleration)
- SSD storage

Environment Setup
-----------------

1. Create Virtual Environment:

   .. code-block:: bash
      :caption: Windows

      python -m venv venv
      venv\Scripts\activate

   .. code-block:: bash
      :caption: macOS/Linux

      python3 -m venv venv
      source venv/bin/activate

2. Upgrade core tools:

   .. code-block:: bash

      pip install --upgrade pip setuptools wheel

Package Installation
--------------------

Method 1: Using requirements.txt (Recommended)

1. Download the project files including ``requirements.txt``
2. Run:

   .. code-block:: bash

      pip install -r requirements.txt

Method 2: Manual Installation (if requirements.txt unavailable)

Core Dependencies:

.. code-block:: bash

   # Data processing
   pip install numpy==1.23.5 pandas==1.5.3 scipy==1.10.0

   # Machine Learning
   pip install scikit-learn==1.2.2 tensorflow==2.12.0 keras==2.12.0
   pip install imbalanced-learn==0.10.1 joblib==1.2.0

   # Visualization
   pip install matplotlib==3.7.1 seaborn==0.12.2 plotly==5.14.1

   # Time Series Analysis
   pip install statsmodels==0.13.5 pmdarima==2.0.3

Optional Dependencies:

.. code-block:: bash

   # Jupyter Notebook support
   pip install jupyter==1.0.0 ipywidgets==8.0.6

   # Dashboard dependencies
   pip install streamlit==1.22.0

GPU Acceleration Setup (Optional)
---------------------------------

For NVIDIA GPU users:

1. Verify CUDA compatibility:

   .. code-block:: bash

      nvidia-smi

2. Install CUDA Toolkit 11.8 and cuDNN 8.6
3. Install TensorFlow with GPU support:

   .. code-block:: bash

      pip install tensorflow[and-cuda]==2.12.0

4. Verify GPU detection:

   .. code-block:: python

      import tensorflow as tf
      print(tf.config.list_physical_devices('GPU'))

Project Structure
----------------

After installation, your project directory should contain:

::

PowerGridAnomalyDetection/
├── data/                          # Sample datasets
├── models/                        # Pretrained models
│   ├── detector/                  # Anomaly detector model
│   │   ├── detector_model.h5
│   │   └── detector_scaler.joblib
│   └── classifier/                # Fault type classifier
│       ├── classifier_model.joblib
│       ├── classifier_scaler.joblib
│       └── class_names.joblib
├── notebooks/                     # Jupyter notebooks for exploration and experimentation
├── dashboard.py                   # Streamlit application for visualization and interaction
├── requirements.txt               # List of dependencies
└── README.md                      # Project overview and usage instructions


Running the System
------------------

Option 1: Jupyter Notebook (Analysis)

.. code-block:: bash

   jupyter notebook notebooks/PowerGridAnalysis.ipynb

Option 2: Streamlit Dashboard (Visualization)

.. code-block:: bash

   streamlit run dashboard.py

Troubleshooting
---------------

Common Issues and Solutions:

1. **Package Conflicts**:
   - Symptom: Import errors or version warnings
   - Solution: Create fresh virtual environment

2. **GPU Not Detected**:
   - Verify CUDA/cuDNN versions match TensorFlow requirements
   - Check NVIDIA drivers are up-to-date

3. **Memory Errors**:
   - Reduce batch size in model configurations
   - Use ``--no-cache-dir`` with pip if RAM is limited

4. **Simulink Data Import**:
   - Ensure CSV files use consistent timestamp formatting
   - Verify voltage/current columns are properly labeled

Platform-Specific Notes
----------------------

Windows:
- May require Visual C++ Build Tools for some packages
- Use PowerShell for better terminal support

macOS:
- May need Homebrew for some system dependencies:
  ``brew install openssl``

Linux:
- Install system packages first:
  ``sudo apt-get install python3-dev python3-venv``

Getting Help
------------

For additional support:
- Consult the project's ``FAQ.md``
- Open an issue on our GitHub repository
- Contact the project maintainers at maintainer@powergridai.com

Next Steps
----------

- :ref:`quickstart`: Learn how to run your first analysis
- :ref:`configuration`: Customize system parameters
- :ref:`troubleshooting`: Detailed error resolution guide