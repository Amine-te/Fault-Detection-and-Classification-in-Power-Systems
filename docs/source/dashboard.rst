PowerAI Dashboard
=================

PowerAI includes an interactive Streamlit-based dashboard for visualizing power system data, detecting anomalies, and classifying faults. This document provides a comprehensive guide to using the dashboard effectively.

.. figure:: _static/dashboard_full.png
   :alt: PowerAI Dashboard
   :align: center

   PowerAI's interactive dashboard for power system fault detection and analysis

Getting Started
---------------

Launching the Dashboard
~~~~~~~~~~~~~~~~~~~~~~~

To launch the PowerAI dashboard, run the following command in your terminal:

.. code-block:: bash

   python app.py

Alternatively, you can use:

.. code-block:: bash

   streamlit run app.py

The dashboard will start and be accessible at http://localhost:8501 in your web browser.

**Command-line Options:**

You can customize the dashboard launch by passing arguments when running the script, for example:

.. code-block:: bash

   python app.py -- --port 8502       # Use a different port
   python app.py -- --theme dark      # Use dark theme
   python app.py -- --wide            # Use wide mode layout

(If your app supports CLI arguments, otherwise set these options inside the app.py as needed.)

Dashboard Interface
-------------------

The dashboard is divided into four main tabs:

- **Data Import**: Upload and preview power system data
- **Model Import**: Import anomaly detection and fault classification models
- **Analysis & Results**: View detected anomalies and fault classifications
- **Visualization**: Detailed visualizations of signals and anomalies

Data Import Tab
~~~~~~~~~~~~~~~

.. figure:: _static/data_import_tab.png
   :alt: Data Import Tab
   :align: center

This tab allows you to:

- **Upload CSV Data**: The file should contain:

  - Time column (timestamps)
  - Voltage signals (Va, Vb, Vc)
  - Current signals (Ia, Ib, Ic)

- **View Data Preview**: Sample of the uploaded data
- **Visualize Signals**: Interactive plots of voltage and current signals

Data Format Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

Your CSV file should follow this format:

.. code-block:: none

   Time,Va,Vb,Vc,Ia,Ib,Ic
   0.000,220.1,220.3,219.8,5.1,5.2,5.0
   0.001,220.2,220.1,219.9,5.2,5.1,5.1
   ...

Notes:

- Time column should be in seconds
- Voltage columns (Va, Vb, Vc) typically in volts
- Current columns (Ia, Ib, Ic) typically in amperes
- Regular sampling rate is recommended
- No missing values in critical columns

Model Import Tab
~~~~~~~~~~~~~~~~

.. figure:: _static/model_import_tab.png
   :alt: Model Import Tab
   :align: center

In this tab, you can:

- **Import Anomaly Detector:**
  - Upload detector model (.h5 file)
  - Upload detector scaler (joblib file)

- **Import Fault Classifier:**
  - Upload classifier model (joblib file)
  - Upload classifier scaler (joblib file)
  - Upload class names mapping (joblib file)

- **Set Analysis Parameters:**
  - Sequence Length: Window size for anomaly detection
  - Anomaly Threshold Percentile: Sensitivity of anomaly detection
  - Max Merge Gap: Maximum gap between anomalies to merge them

- **Run Analysis**: Process data with the loaded models

File Types Overview
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Component
     - File Type
     - Description
   * - Detector Model
     - .h5
     - TensorFlow LSTM autoencoder model
   * - Detector Scaler
     - .joblib
     - StandardScaler for preprocessing input data
   * - Classifier Model
     - .joblib
     - Trained machine learning classifier
   * - Classifier Scaler
     - .joblib
     - Scaler for feature preprocessing
   * - Class Names
     - .joblib
     - Dictionary mapping class indices to fault names

Analysis & Results Tab
~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/analysis_tab.png
   :alt: Analysis Tab
   :align: center

After running the analysis, this tab shows:

- **Summary Metrics**:
  - Total number of detected anomalies
  - Number of unique fault types
  - Average anomaly duration

- **Detected Anomalies Table**:
  - Start and end times of each anomaly
  - Predicted fault type
  - Confidence level for each prediction
  - Duration of each anomaly

- **Fault Type Distribution**:
  - Bar chart showing the distribution of detected fault types

- **Anomaly Timeline**:
  - Timeline visualization of all detected anomalies

Understanding Classification Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fault classification results include:

- Predicted Fault: The type of fault detected
- Confidence: Probability estimate for the prediction
- Duration: Time span of the anomaly
- Start/End Time: Precise timing of the fault

Visualization Tab
~~~~~~~~~~~~~~~~~

.. figure:: _static/visualization_tab.png
   :alt: Visualization Tab
   :align: center

This tab offers three visualization types:

- **Signal Visualization with Anomalies**:
  - Selected signals with highlighted anomaly regions
  - Annotations for each detected fault

- **Reconstruction Error**:
  - Plot of reconstruction error over time
  - Threshold line and highlighted anomalies
  - Histogram of reconstruction error distribution

- **Feature Importance**:
  - Bar chart of top features for fault classification
  - Feature importance by signal type (pie chart)

Export Options
--------------

The dashboard provides several export options:

- **Download Results as CSV**:
  - Complete table of all detected anomalies

- **Download Summary Report**:
  - Markdown report with key findings
  - Summary statistics
  - Fault type distribution

Using the Dashboard in Production
---------------------------------

For production use, consider:

- Setting up the dashboard on a dedicated server
- Configuring authentication for restricted access
- Setting up automated data ingestion
- Scheduling regular analyses
- Integrating with notification systems

Advanced Features
-----------------

Custom Thresholds
~~~~~~~~~~~~~~~~~

Fine-tune anomaly detection by adjusting the threshold percentile:

- Higher values (e.g., 99th percentile): Fewer anomalies, higher confidence
- Lower values (e.g., 90th percentile): More sensitive detection, potential false positives

Signal Selection
~~~~~~~~~~~~~~~~

Select specific signals to focus your analysis:

- Voltage signals (Va, Vb, Vc) for voltage-related issues
- Current signals (Ia, Ib, Ic) for current-related faults
- Combined analysis for comprehensive detection

Interactive Exploration
~~~~~~~~~~~~~~~~~~~~~~~

The dashboard supports interactive exploration:

- Zoom in/out of signal plots
- Hover over data points for detailed information
- Filter anomalies by fault type
- Sort results by various criteria

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

- **Dashboard Not Loading**:
  - Check if Streamlit is installed correctly
  - Verify port 8501 is available
  - Check for Python environment issues

- **File Upload Errors**:
  - Ensure CSV format is correct
  - Check file size (limit: 200MB)
  - Verify column names match requirements

- **Model Import Failures**:
  - Verify model file formats (.h5, .joblib)
  - Check if models were trained with compatible libraries
  - Confirm models are not corrupted

- **No Anomalies Detected**:
  - Try lowering the threshold percentile
  - Verify data contains actual anomalies
  - Check if models are appropriate for your system

Performance Tips
----------------

For large datasets:

- Use data sampling for faster visualization
- Run analysis on specific time segments
- Consider preprocessing data before upload
- Use hardware acceleration when available

Next Steps
----------

- :doc:`usage` - See examples of practical applications
- :doc:`api` - Explore programmatic access to PowerAI
- :doc:`models` - Learn more about the underlying models
