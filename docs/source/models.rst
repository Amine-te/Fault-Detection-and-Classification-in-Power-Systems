=============
Models
=============

PowerAI uses a two-stage approach to detect and classify faults in power systems:

- **Anomaly Detection**: An LSTM-based autoencoder identifies abnormal patterns in power system signals.
- **Fault Classification**: A machine learning classifier categorizes detected anomalies into specific fault types.

This document explains the architecture, training, and usage of these models.

.. figure:: _static/model_architecture.png
   :alt: PowerAI Model Architecture
   :align: center

   Two-stage model architecture for power system fault detection and classification

Anomaly Detection Model
=======================

Architecture
------------

The anomaly detection system uses a Long Short-Term Memory (LSTM) autoencoder neural network with the following architecture:

.. code-block:: none

   Input Layer (shape: [batch_size, sequence_length, 6])
   ↓
   LSTM Encoder (units=64, return_sequences=True)
   ↓
   LSTM Encoder (units=32, return_sequences=False)
   ↓
   RepeatVector (sequence_length)
   ↓
   LSTM Decoder (units=32, return_sequences=True)
   ↓
   LSTM Decoder (units=64, return_sequences=True)
   ↓
   Time Distributed Dense (units=6, activation='linear')
   ↓
   Output Layer (shape: [batch_size, sequence_length, 6])

The model takes sequences of voltage and current measurements (Va, Vb, Vc, Ia, Ib, Ic) and attempts to reconstruct them. Anomalies are detected when the reconstruction error exceeds a threshold.

Training Process
----------------

**Data Preprocessing:**

- Normalization using StandardScaler
- Sequence creation with a sliding window approach
- Train-validation split (typically 80%-20%)

**Training Parameters:**

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate of 0.001
- Epochs: 100 (with early stopping)
- Batch size: 32

**Threshold Selection:**

- Reconstruction errors are calculated on validation data
- Threshold is set at a high percentile (e.g., 95th or 99th) of these errors

Inference and Anomaly Detection
-------------------------------

During inference:

- New data is preprocessed and fed through the autoencoder
- Reconstruction error (MSE) is calculated for each window
- Windows with errors above the threshold are flagged as anomalies
- Consecutive or close anomalies are merged into events

Fault Classification Model
==========================

Architecture
------------

The fault classifier is a machine learning model (typically a Random Forest or Gradient Boosting classifier) that takes extracted features from detected anomalies and classifies them into specific fault types.

Feature Extraction
------------------

For each detected anomaly, the following features are extracted:

**Time-domain features:**

- Mean, standard deviation, skewness, kurtosis
- RMS values, peak-to-peak amplitude
- Crest factor

**Cross-signal features:**

- Correlations between voltage and current phases
- Phase imbalance metrics

**Duration features:**

- Anomaly duration
- Rise/fall times

These features are combined into a feature vector for classification.

Supported Fault Types
---------------------

The classifier is trained to recognize various power system faults, including:

- Single-phase to ground faults
- Phase-to-phase faults
- Double-phase to ground faults
- Three-phase faults

Training Process
----------------

**Data Collection:**

- Simulated fault data from power system simulators
- Real-world fault recordings (if available)
- Augmented data to increase robustness

**Feature Engineering:**

- Calculate all features for each fault event
- Feature selection to identify most discriminative features
- Feature scaling using StandardScaler

**Model Training:**

- Cross-validation (k-fold)
- Hyperparameter optimization
- Class weighting for imbalanced fault types

Model Evaluation
================

Performance Metrics
-------------------

**Anomaly Detector:**

- Precision, Recall, F1-Score
- Detection latency
- False positive rate

**Fault Classifier:**

- Classification accuracy
- Confusion matrix
- Per-class precision/recall

Benchmark Results
-----------------

On standard IEEE test systems:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - Model
     - Metric
     - Value
     - Comparison
   * - Anomaly Detector
     - F1-Score
     - 0.95
     - +12% vs traditional methods
   * - Anomaly Detector
     - Detection Latency
     - 50ms
     - -70% vs rule-based systems
   * - Fault Classifier
     - Accuracy
     - 0.92
     - +8% vs conventional classifiers
   * - Fault Classifier
     - Mean Confidence
     - 0.88
     - +15% reliability improvement

Model File Structure
====================

The PowerAI models are stored in the following format:

.. code-block:: none

   models/
   ├── detector/
   │   ├── detector_model.h5         # TensorFlow LSTM autoencoder model
   │   └── detector_scaler.joblib    # StandardScaler for input preprocessing
   │
   └── classifier/
       ├── classifier_model.joblib   # Trained classifier model
       ├── classifier_scaler.joblib  # Feature scaler
       └── class_names.joblib        # Mapping of numerical labels to fault types

Custom Models
=============

Creating Custom Models
----------------------

You can train custom models using your own power system data:

.. code-block:: python

   from powerai.models import train_detector, train_classifier

   # Train anomaly detector
   detector_model, detector_scaler = train_detector(
       data_path="path/to/normal_data.csv",
       sequence_length=50,
       epochs=100
   )

   # Train fault classifier
   classifier_model, classifier_scaler, class_names = train_classifier(
       data_path="path/to/fault_data.csv",
       feature_selection=True
   )

   # Save models
   from powerai.utils import save_models

   save_models(
       detector_model, detector_scaler,
       classifier_model, classifier_scaler, class_names,
       output_dir="custom_models/"
   )

Model Customization Options
---------------------------

**Detector Customization:**

- Sequence length (window size)
- Number of LSTM layers and units
- Anomaly threshold percentile

**Classifier Customization:**

- Feature selection method
- Classifier algorithm (Random Forest, XGBoost, etc.)
- Custom fault types

Next Steps
==========

- :doc:`dashboard` - Learn how to use the interactive dashboard with these models
