Power System Anomaly Dashboard
==============================

Introduction
------------

The Power System Anomaly Dashboard is an interactive web application built with Streamlit that provides comprehensive anomaly detection and classification for power system data. It allows engineers and analysts to identify, analyze, and categorize faults in electrical power systems through a user-friendly interface.

.. image:: https://via.placeholder.com/800x400.png
   :alt: Power System Anomaly Dashboard
   :align: center

Features
--------

- **Data Importing**: Upload and visualize power system data in CSV format
- **Model Integration**: Import pre-trained anomaly detection and classification models
- **Anomaly Detection**: Identify anomalous events in power signal data
- **Fault Classification**: Categorize detected anomalies into specific fault types
- **Interactive Visualization**: Explore signals, anomalies, and analysis results through dynamic charts
- **Reporting**: Export analysis results and generate summary reports

System Requirements
-------------------

- Python 3.7+
- Required packages:
  
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `joblib`
  - `tensorflow`
  - `scipy`
  - `plotly`

Installation
------------

1. Clone the repository or download the dashboard script.
2. Install the required packages:

::

   pip install streamlit pandas numpy matplotlib seaborn joblib tensorflow scipy plotly

3. Launch the dashboard:

::

   streamlit run dashboard.py

Input Data Format
-----------------

The dashboard requires CSV data with the following columns:

- **Time**: Timestamps for the signals
- **Voltage signals**: Va, Vb, Vc (three-phase voltage measurements)
- **Current signals**: Ia, Ib, Ic (three-phase current measurements)

Required Models
---------------

The dashboard requires pre-trained models for operation:

1. **Anomaly Detector**:

   - LSTM Autoencoder model (`.h5` file)
   - `StandardScaler` (`.joblib` file)

2. **Fault Classifier**:

   - Classification model (`.joblib` file)
   - `StandardScaler` (`.joblib` file)
   - Class names mapping (`.joblib` file)

User Guide
----------

### Step 1: Data Import

1. Navigate to the "Data Import" tab.
2. Upload your power system CSV data.
3. Preview and visualize the data.
4. Ensure the following columns are present: Time, Va, Vb, Vc, Ia, Ib, Ic.

### Step 2: Model Import

1. Go to the "Model Import" tab.
2. Upload:
   - Autoencoder model (`.h5`) and scaler.
   - Classifier model (`.joblib`), scaler, and class names.
3. Configure:
   - Sequence Length (default: 50)
   - Anomaly Threshold Percentile (default: 95%)
   - Max Merge Gap (default: 1500 samples)
4. Click **Run Analysis**.

### Step 3: Analysis & Results

- View metrics: total anomalies, fault types, average anomaly duration.
- Review detected anomaly intervals.
- Explore visual analytics (distribution, timeline, etc.).

### Step 4: Visualization

Choose from:

- **Signal Visualization with Anomalies**
- **Reconstruction Error**
- **Feature Importance**

Export Options
--------------

- **Download Results as CSV**
- **Download Summary Report** (Markdown format)

Technical Details
-----------------

### Anomaly Detection Approach

1. **Preprocessing**:
   - Normalize (StandardScaler)
   - Create overlapping sequences

2. **Autoencoder**:
   - LSTM autoencoder detects reconstruction error
   - Threshold using percentile

3. **Merging**:
   - Merge intervals using `merge_intervals`

4. **Feature Extraction**:
   - Mean, std, skew, kurtosis, RMS, crest factor, correlations

5. **Classification**:
   - Normalize features
   - Classify using trained model
   - Confidence score (if applicable)

Functions Reference
-------------------

**extract_features(data, interval)**

- Extracts statistical and signal-based features.
- Returns a feature dictionary.

**preprocess_data(data, scaler, sequence_length)**

- Normalizes and sequences data for autoencoder input.
- Returns a numpy array: `(n_sequences, sequence_length, n_features)`

**merge_intervals(intervals, max_gap_samples, sample_time)**

- Merges nearby intervals.
- Returns a list of merged `(start_time, end_time)` tuples.

**detect_anomalies(data, detector_model, detector_scaler, sequence_length, percentile)**

- Returns anomaly flags, reconstruction error, window timestamps, intervals, and threshold.

**predict_anomaly_intervals(data, interval_list, classifier_model, classifier_scaler, class_names)**

- Classifies intervals into fault types.
- Returns a DataFrame with predictions and confidence.

Customization
-------------

You can:

- Change sequence length and thresholds
- Modify `extract_features` for new features
- Enhance visualizations via Streamlit
- Adjust UI components

Troubleshooting
---------------

**Common Issues**:

- **Missing Columns**: Ensure data has required columns.
- **Model Load Failures**: Check model and scaler file compatibility.
- **Memory Errors**: Use smaller files or downsample data.
- **No Anomalies Detected**: Lower the threshold.

Performance Optimization
------------------------

- Automatic downsampling for large files
- Tips:
  - Analyze smaller periods
  - Tune sequence length
  - Use higher-spec machines

References
----------

- `Streamlit <https://streamlit.io/>`_
- `TensorFlow <https://www.tensorflow.org/>`_
- `Plotly <https://plotly.com/>`_
- `Joblib <https://joblib.readthedocs.io/>`_
