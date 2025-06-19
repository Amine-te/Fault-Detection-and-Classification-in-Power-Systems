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
- Git (for repository cloning)

Recommended for optimal performance:
- Python 3.10
- 16GB+ RAM
- NVIDIA GPU with CUDA support (for TensorFlow acceleration)
- SSD storage

Getting the Project
-------------------

1. Clone the Git Repository:

   .. code-block:: bash

      git clone https://github.com/Amine-te/Fault-Detection-and-Classification-in-Power-Systems
      cd PowerGridAnomalyDetection

   Alternative using SSH:

   .. code-block:: bash

      git clone git@github.com:Amine-te/Fault-Detection-and-Classification-in-Power-Systems
      cd PowerGridAnomalyDetection

2. Verify the project structure:

   .. code-block:: bash

      ls -la

   You should see the main project files and directories.

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

1. Install all dependencies from the cloned repository:

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

After cloning and installation, your project directory should contain:

::

   PowerGridAnomalyDetection/
   ├── .git/                          # Git repository metadata
   ├── .gitignore                     # Git ignore file
   ├── data/                          # Sample datasets
   ├── models/                        # Pretrained models
   │   ├── detector/                  # Anomaly detector model
   │   │   ├── detector_model.h5
   │   │   └── detector_scaler.joblib
   │   ├── classifier/                # Fault type classifier
   │   │   ├── classifier_model.joblib
   │   │   ├── classifier_scaler.joblib
   │   │   └── class_names.joblib
   │   └── classifier_transformer/    # Advanced deep learning fault classifier
   │       ├── best_fault_classifier.h5
   │       ├── complete_dl_fault_system.joblib
   │       ├── feature_scaler.joblib
   │       ├── label_encoder.joblib
   │       ├── model_metadata.joblib
   │       └── sequence_scaler.joblib
   ├── TimSeriesProject.ipynb         # Jupyter notebooks for exploration and experimentation
   ├── app.py                         # Streamlit application for visualization and interaction
   ├── nlpQueryinterface.py           # NLP query interface
   ├── requirements.txt               # List of dependencies
   ├── README.md                      # Project overview and usage instructions
   └── LICENSE                        # Project license file


Git Repository Management
------------------------

Useful Git commands for development:

.. code-block:: bash

   # Check repository status
   git status

   # Pull latest changes
   git pull origin main

   # Create a new branch for development
   git checkout -b feature/your-feature-name

   # Stage and commit changes
   git add .
   git commit -m "Your commit message"

   # Push changes to remote repository
   git push origin your-branch-name

Running the System
------------------

Option 1: Jupyter Notebook (Analysis)

.. code-block:: bash

   jupyter notebook TimSeriesProject.ipynb

Option 2: Streamlit Dashboard (Visualization)

.. code-block:: bash

   streamlit run app.py

Troubleshooting
---------------

Common Issues and Solutions:

1. **Git Clone Issues**:
   - Symptom: Permission denied or repository not found
   - Solution: Check repository URL and authentication credentials
   - For private repos: Use personal access token or SSH key

2. **Package Conflicts**:
   - Symptom: Import errors or version warnings
   - Solution: Create fresh virtual environment

3. **GPU Not Detected**:
   - Verify CUDA/cuDNN versions match TensorFlow requirements
   - Check NVIDIA drivers are up-to-date

4. **Memory Errors**:
   - Reduce batch size in model configurations
   - Use ``--no-cache-dir`` with pip if RAM is limited

5. **Simulink Data Import**:
   - Ensure CSV files use consistent timestamp formatting
   - Verify voltage/current columns are properly labeled

Platform-Specific Notes
----------------------

Windows:
- May require Visual C++ Build Tools for some packages
- Use PowerShell for better terminal support
- Install Git for Windows if not available

macOS:
- May need Homebrew for some system dependencies:
  ``brew install openssl git``
- Git usually comes pre-installed with Xcode Command Line Tools

Linux:
- Install system packages first:
  ``sudo apt-get install python3-dev python3-venv git``

Getting Help
------------

For additional support:
- Consult the project's ``FAQ.md``
- Open an issue on our GitHub repository
- Check the commit history for recent changes: ``git log --oneline``