=============================
Power System Anomaly Dashboard
Introduction
The Power System Anomaly Dashboard is an interactive web application built with Streamlit that provides comprehensive anomaly detection and classification for power system data. It allows engineers and analysts to identify, analyze, and categorize faults in electrical power systems through a user-friendly interface.
.. image:: /api/placeholder/800/400
:alt: Power System Anomaly Dashboard
:align: center
Features

Data Importing: Upload and visualize power system data in CSV format
Model Integration: Import pre-trained anomaly detection and classification models
Anomaly Detection: Identify anomalous events in power signal data
Fault Classification: Categorize detected anomalies into specific fault types
Interactive Visualization: Explore signals, anomalies, and analysis results through dynamic charts
Reporting: Export analysis results and generate summary reports

System Requirements

Python 3.7+
Required packages:

streamlit
pandas
numpy
matplotlib
seaborn
joblib
tensorflow
scipy
plotly



Installation

Clone the repository or download the dashboard script
Install the required packages:

.. code-block:: bash
pip install streamlit pandas numpy matplotlib seaborn joblib tensorflow scipy plotly

Launch the dashboard:

.. code-block:: bash
streamlit run dashboard.py
Input Data Format
The dashboard requires CSV data with the following columns:

Time: Timestamps for the signals
Voltage signals: Va, Vb, Vc (three-phase voltage measurements)
Current signals: Ia, Ib, Ic (three-phase current measurements)

Required Models
The dashboard requires pre-trained models for operation:

Anomaly Detector:

LSTM Autoencoder model (.h5 file) for detecting anomalies
StandardScaler (.joblib file) for preprocessing data


Fault Classifier:

Classification model (.joblib file) for categorizing fault types
StandardScaler (.joblib file) for feature preprocessing
Class names mapping (.joblib file) for fault type labels



User Guide
Step 1: Data Import

Navigate to the "Data Import" tab
Upload your power system CSV data using the file uploader
Verify the data preview and visualize voltage and current signals
Ensure your data has the required Time, Va, Vb, Vc, Ia, Ib, Ic columns

Step 2: Model Import

Navigate to the "Model Import" tab
Upload the required models:

Detector model (.h5) and associated scaler
Classifier model (.joblib), scaler, and class names mapping


Configure analysis parameters:

Sequence Length: Window size for anomaly detection (default: 50)
Anomaly Threshold Percentile: Sensitivity of anomaly detection (default: 95%)
Max Merge Gap: Maximum separation between anomalies to merge them (default: 1500 samples)


Click "Run Analysis" to process the data

Step 3: Analysis & Results

View the analysis results in the "Analysis & Results" tab
Examine metrics like total anomalies, unique fault types, and average anomaly duration
Review the table of detected anomalies with their timestamps, durations, and classifications
Explore visualizations showing fault type distribution and anomaly timeline

Step 4: Visualization

Navigate to the "Visualization" tab
Choose visualization types:

Signal Visualization with Anomalies: View original signals with highlighted anomaly regions
Reconstruction Error: Analyze autoencoder reconstruction errors and detection threshold
Feature Importance: Explore which features are most significant for classification


Interact with the visualizations to gain insights about the detected anomalies

Export Options

Download Results as CSV: Export the full table of detected anomalies
Download Summary Report: Generate a markdown report with key statistics and findings

Technical Details
Anomaly Detection Approach
The dashboard employs a multi-stage approach for detecting and classifying power system anomalies:

Signal Preprocessing:

The input signals (Va, Vb, Vc, Ia, Ib, Ic) are normalized using a StandardScaler
Time-series data is segmented into overlapping sequences for the autoencoder


Autoencoder-Based Detection:

An LSTM autoencoder model reconstructs normal signal patterns
Higher reconstruction errors indicate potential anomalies
A percentile-based threshold determines anomaly classification


Interval Merging:

Consecutive or nearby anomaly windows are merged into larger events
The merge_intervals function handles this with a configurable maximum gap parameter


Feature Extraction:

For each detected anomaly interval, features are extracted from the raw signals
Features include statistical measures (mean, std, skew, kurtosis) and signal characteristics (RMS, peak-to-peak, crest factor)
Cross-signal correlations are calculated to capture phase relationships


Fault Classification:

The extracted features are normalized and fed to the classification model
The model predicts the specific fault type for each anomaly interval
Confidence scores are provided when available (for probabilistic classifiers)



Functions Reference
extract_features(data, interval)
Extracts features from raw signal data for a given time interval.
Parameters:

data: DataFrame containing the power system signals
interval: Tuple (start_time, end_time) defining the interval

Returns:

Dictionary of features including statistical measures and signal characteristics

preprocess_data(data, scaler, sequence_length)
Prepares data for the autoencoder by normalizing and creating overlapping sequences.
Parameters:

data: DataFrame containing the power system signals
scaler: Trained StandardScaler for normalizing the data
sequence_length: Number of samples in each sequence

Returns:

Numpy array of shape (n_sequences, sequence_length, n_features)

merge_intervals(intervals, max_gap_samples, sample_time)
Merges anomaly intervals that are close to each other in time.
Parameters:

intervals: List of (start_time, end_time) tuples
max_gap_samples: Maximum number of samples between intervals to merge
sample_time: Time difference between consecutive samples

Returns:

List of merged (start_time, end_time) intervals

detect_anomalies(data, detector_model, detector_scaler, sequence_length, percentile)
Detects anomalies in the power system data using an autoencoder.
Parameters:

data: DataFrame containing the power system signals
detector_model: Trained autoencoder model
detector_scaler: Trained StandardScaler for preprocessing
sequence_length: Number of samples in each sequence
percentile: Percentile threshold for anomaly detection

Returns:

Boolean array of anomaly flags
Array of reconstruction error scores
Array of timestamps corresponding to each window
List of original anomaly intervals
Threshold value

predict_anomaly_intervals(data, interval_list, classifier_model, classifier_scaler, class_names)
Classifies detected anomaly intervals into specific fault types.
Parameters:

data: DataFrame containing the power system signals
interval_list: List of (start_time, end_time) intervals
classifier_model: Trained classifier model
classifier_scaler: Trained StandardScaler for feature preprocessing
class_names: Mapping from numeric indices to fault type names

Returns:

DataFrame with columns: start_time, end_time, predicted_fault, confidence

Customization
The dashboard includes several customizable parameters:

Sequence Length: Adjusts the window size for anomaly detection (impacts sensitivity to short anomalies)
Anomaly Threshold Percentile: Controls the sensitivity of the anomaly detector (higher = fewer anomalies)
Max Merge Gap: Determines how close anomalies must be to be considered part of the same event

Advanced users can modify the dashboard code to:

Add new feature extraction methods in the extract_features function
Implement additional visualization options
Customize the appearance and layout using Streamlit's UI components

Troubleshooting
Common Issues

Data format errors:

Ensure your CSV file contains all required columns: Time, Va, Vb, Vc, Ia, Ib, Ic
Check for missing values or inconsistent formatting


Model loading failures:

Verify that model files are compatible with the dashboard
Ensure all necessary files are uploaded (model, scaler, class names)


Memory errors:

For large datasets, consider downsampling or analyzing smaller segments
Increase system memory allocation for Streamlit


No anomalies detected:

Try adjusting the anomaly threshold percentile to a lower value
Verify that the data contains actual anomalies or faults



Performance Optimization

For large datasets, the dashboard automatically downsamples data for visualization
To improve performance further, consider:

Using a more powerful machine for running the dashboard
Pre-filtering data to focus on time periods of interest
Optimizing the sequence length parameter for your specific use case



References

Streamlit: https://streamlit.io/
TensorFlow: https://www.tensorflow.org/
Plotly: https://plotly.com/
Joblib: https://joblib.readthedocs.io/

License
[Specify license information here]
Contact
[Provide contact information for support or contributions]