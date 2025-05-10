import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from scipy.stats import skew, kurtosis
import os
import tempfile
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Power System Anomaly Dashboard",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        margin-right: 2px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5 !important;
        color: white !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    div.stButton > button:first-child {
        background-color: #4e8df5;
        color: white;
        font-weight: 500;
    }
    div.stButton > button:hover {
        background-color: #3a7bd5;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 16px;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("⚡ Power System Anomaly Detection Dashboard")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'detector_model' not in st.session_state:
    st.session_state.detector_model = None
if 'detector_scaler' not in st.session_state:
    st.session_state.detector_scaler = None
if 'classifier_model' not in st.session_state:
    st.session_state.classifier_model = None
if 'classifier_scaler' not in st.session_state:
    st.session_state.classifier_scaler = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'merged_intervals' not in st.session_state:
    st.session_state.merged_intervals = None
if 'mse_scores' not in st.session_state:
    st.session_state.mse_scores = None
if 'anomaly_times' not in st.session_state:
    st.session_state.anomaly_times = None

# Create a function for feature extraction
def extract_features(data, interval):
    """Extract features from raw data for a given time interval."""
    start_idx = np.searchsorted(data['Time'], interval[0])
    end_idx = np.searchsorted(data['Time'], interval[1])
    window = data.iloc[start_idx:end_idx]
    
    features = {
        'duration': interval[1] - interval[0]  # Direct duration calculation
    }
    
    # Signal features
    for signal in ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']:
        sig_data = window[signal].values
        features.update({
            f'{signal}_mean': np.mean(sig_data),
            f'{signal}_std': np.std(sig_data),
            f'{signal}_skew': skew(sig_data),
            f'{signal}_kurt': kurtosis(sig_data),
            f'{signal}_rms': np.sqrt(np.mean(sig_data**2)),
            f'{signal}_peak2peak': np.ptp(sig_data),
            f'{signal}_crest': np.max(np.abs(sig_data)) / (np.sqrt(np.mean(sig_data**2)) + 1e-9)  # Avoid division by zero
        })
    
    # Cross-signal correlations
    features.update({
        'corr_VaVb': np.corrcoef(window['Va'], window['Vb'])[0,1],
        'corr_VbVc': np.corrcoef(window['Vb'], window['Vc'])[0,1],
        'corr_VaIa': np.corrcoef(window['Va'], window['Ia'])[0,1]
    })
    
    return features

# Function to preprocess data for the autoencoder
def preprocess_data(data, scaler, sequence_length):
    # Normalize using the scaler
    scaled_data = scaler.transform(data[['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']])
    
    # Create sequences
    X = []
    for i in range(len(scaled_data) - sequence_length + 1):
        X.append(scaled_data[i:i+sequence_length])
    
    return np.array(X)

# Function to merge time intervals that are close to each other
def merge_intervals(intervals, max_gap_samples=1500, sample_time=None):
    if not intervals:
        return []

    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [list(sorted_intervals[0])]
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        # Calculate max gap time based on sample time
        max_gap_time = max_gap_samples * sample_time if sample_time else 0.01
        
        # Check gap between last interval's end and current interval's start
        if current_start <= last_end + max_gap_time:
            # Merge intervals
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])
    
    return [tuple(interval) for interval in merged]

# Function to detect anomalies using the autoencoder
def detect_anomalies(data, detector_model, detector_scaler, sequence_length=50, percentile=95):
    # Preprocess data
    X_sequences = preprocess_data(data, detector_scaler, sequence_length)
    
    # Get reconstruction errors
    reconstructions = detector_model.predict(X_sequences)
    mse = np.mean(np.square(X_sequences - reconstructions), axis=(1,2))
    
    # Set threshold (e.g., 95th percentile)
    threshold = np.percentile(mse, percentile)
    
    # Find anomalies
    anomalies = mse > threshold
    
    # Get timestamps (aligned with window end times)
    time_values = data['Time'].values[sequence_length - 1:]
    
    # Get anomaly window indices
    anomaly_window_indices = np.where(anomalies)[0]
    
    # Convert window indices to original data time intervals
    original_intervals = []
    for win_idx in anomaly_window_indices:
        start_idx = win_idx
        end_idx = win_idx + sequence_length - 1
        if end_idx < len(data):
            start_time = data['Time'].iloc[start_idx]
            end_time = data['Time'].iloc[end_idx]
            original_intervals.append((start_time, end_time))
    
    return anomalies, mse, time_values, original_intervals, threshold

# Function to predict anomaly types
def predict_anomaly_intervals(data, interval_list, classifier_model, classifier_scaler, class_names):
    """
    Predict fault types for anomaly intervals
    """
    if not interval_list:
        return pd.DataFrame({
            'start_time': [],
            'end_time': [],
            'predicted_fault': [],
            'confidence': []
        })
    
    # Feature extraction
    features = []
    for interval in interval_list:
        features.append(extract_features(data, interval))
    X_new = pd.DataFrame(features)
    
    # Preprocessing
    X_new_scaled = classifier_scaler.transform(X_new)
    
    # Predictions
    preds = classifier_model.predict(X_new_scaled)
    
    # Handle different formats of class_names
    if isinstance(class_names, (list, np.ndarray)):
        predicted_labels = [class_names[i] if isinstance(i, (int, np.integer)) else i for i in preds]
    elif isinstance(class_names, dict):
        predicted_labels = [class_names.get(i, str(i)) for i in preds]
    else:
        predicted_labels = [str(p) for p in preds]
    
    # Confidence scores (if supported)
    if hasattr(classifier_model, 'predict_proba'):
        probas = classifier_model.predict_proba(X_new_scaled)
        confidences = np.max(probas, axis=1)
    else:  # For models without probability estimates
        confidences = np.ones(len(preds)) * np.nan
    
    return pd.DataFrame({
        'start_time': [i[0] for i in interval_list],
        'end_time': [i[1] for i in interval_list],
        'predicted_fault': predicted_labels,
        'confidence': confidences
    })

# Format predictions for display
def format_predictions(predictions_df):
    if predictions_df.empty:
        return predictions_df
    
    # Add duration column
    predictions_df['duration'] = predictions_df['end_time'] - predictions_df['start_time']
    
    # Format confidence
    predictions_df['confidence'] = predictions_df['confidence'].apply(
        lambda x: f"{x:.1%}" if not np.isnan(x) else 'N/A')
    
    # Sort by start time
    return predictions_df.sort_values('start_time').reset_index(drop=True)

# Main application layout with tabs
tabs = st.tabs(["Data Import", "Model Import", "Analysis & Results", "Visualization"])

# Tab 1: Data Import
with tabs[0]:
    st.header("Step 1: Import Power System Data")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV Data", type="csv", 
                                       help="Upload a CSV file with columns: Time, Va, Vb, Vc, Ia, Ib, Ic")
        
        if uploaded_file is not None:
            try:
                # Read data
                data = pd.read_csv(uploaded_file)
                
                # Validate data format
                required_cols = ['Time', 'Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
                if all(col in data.columns for col in required_cols):
                    st.session_state.data = data
                    st.success(f"✅ Data loaded successfully! ({len(data)} rows)")
                else:
                    st.error("❌ Data format incorrect. Please ensure your CSV has columns: Time, Va, Vb, Vc, Ia, Ib, Ic")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.subheader("Data Format Requirements")
        st.info("""
        Your CSV file should contain:
        - Time column (timestamps)
        - Voltage readings (Va, Vb, Vc)
        - Current readings (Ia, Ib, Ic)
        """)
        
        if st.session_state.data is not None:
            st.metric("Number of Samples", f"{len(st.session_state.data):,}")
            time_range = st.session_state.data['Time'].max() - st.session_state.data['Time'].min()
            st.metric("Time Range", f"{time_range:.2f} s")
    
    # Show data preview if available
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Quick signal visualization
        st.subheader("Signal Visualization")
        
        # Sample data for faster plotting if dataset is large
        sample_data = st.session_state.data
        if len(sample_data) > 5000:
            sample_step = max(1, len(sample_data) // 5000)
            sample_data = sample_data.iloc[::sample_step].reset_index(drop=True)
        
        # Separate visualizations for voltage and current
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Voltage Signals")
            voltage_signals = st.multiselect(
                "Select voltage signals:", 
                ['Va', 'Vb', 'Vc'],
                default=['Va', 'Vb', 'Vc']
            )
            
            if voltage_signals:
                fig_voltage = px.line(sample_data, x='Time', y=voltage_signals, 
                              title='Voltage Signals',
                              labels={'value': 'Voltage (V)', 'variable': 'Signal'})
                
                fig_voltage.update_layout(
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig_voltage, use_container_width=True)
        
        with col2:
            st.markdown("### Current Signals")
            current_signals = st.multiselect(
                "Select current signals:", 
                ['Ia', 'Ib', 'Ic'],
                default=['Ia', 'Ib', 'Ic']
            )
            
            if current_signals:
                fig_current = px.line(sample_data, x='Time', y=current_signals, 
                              title='Current Signals',
                              labels={'value': 'Current (A)', 'variable': 'Signal'})
                
                fig_current.update_layout(
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig_current, use_container_width=True)

# Tab 2: Model Import
with tabs[1]:
    st.header("Step 2: Import Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anomaly Detector")
        
        detector_model_file = st.file_uploader("Upload Detector Model (.h5)", type=['h5'], 
                                              help="Upload the LSTM autoencoder model file")
        
        detector_scaler_file = st.file_uploader("Upload Detector Scaler", type=['joblib', 'save'], 
                                              help="Upload the scaler used for preprocessing")
        
        if detector_model_file and detector_scaler_file:
            try:
                # Save the uploaded files to temporary files
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_model:
                    tmp_model.write(detector_model_file.getvalue())
                    detector_model_path = tmp_model.name
                
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_scaler:
                    tmp_scaler.write(detector_scaler_file.getvalue())
                    detector_scaler_path = tmp_scaler.name
                
                # Load the saved files
                st.session_state.detector_model = tf.keras.models.load_model(
                    detector_model_path, 
                    custom_objects={'mse': 'mse'}
                )
                st.session_state.detector_scaler = joblib.load(detector_scaler_path)
                
                # Clean up temporary files
                os.unlink(detector_model_path)
                os.unlink(detector_scaler_path)
                
                st.success("✅ Detector model and scaler loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading detector model: {str(e)}")
    
    with col2:
        st.subheader("Fault Classifier")
        
        classifier_model_file = st.file_uploader("Upload Classifier Model", type=['joblib'], 
                                              help="Upload the fault classifier model")
        
        classifier_scaler_file = st.file_uploader("Upload Classifier Scaler", type=['joblib'], 
                                                help="Upload the scaler for classifier features")
        
        class_names_file = st.file_uploader("Upload Class Names", type=['joblib'], 
                                          help="Upload the class names mapping")
        
        if classifier_model_file and classifier_scaler_file and class_names_file:
            try:
                # Save the uploaded files to temporary files
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_model:
                    tmp_model.write(classifier_model_file.getvalue())
                    classifier_model_path = tmp_model.name
                
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_scaler:
                    tmp_scaler.write(classifier_scaler_file.getvalue())
                    classifier_scaler_path = tmp_scaler.name
                
                with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_classes:
                    tmp_classes.write(class_names_file.getvalue())
                    class_names_path = tmp_classes.name
                
                # Load the saved files
                st.session_state.classifier_model = joblib.load(classifier_model_path)
                st.session_state.classifier_scaler = joblib.load(classifier_scaler_path)
                st.session_state.class_names = joblib.load(class_names_path)
                
                # Clean up temporary files
                os.unlink(classifier_model_path)
                os.unlink(classifier_scaler_path)
                os.unlink(class_names_path)
                
                st.success("✅ Classifier model, scaler, and class names loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading classifier model: {str(e)}")
    
    # Display model parameters if models are loaded
    if (st.session_state.data is not None and 
        st.session_state.detector_model is not None and 
        st.session_state.detector_scaler is not None and
        st.session_state.classifier_model is not None and
        st.session_state.classifier_scaler is not None and
        st.session_state.class_names is not None):
        
        st.subheader("Analysis Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sequence_length = st.slider("Sequence Length", min_value=10, max_value=100, value=50,
                                      help="Window size for anomaly detection")
        
        with col2:
            anomaly_percentile = st.slider("Anomaly Threshold Percentile", min_value=90, max_value=99, value=95,
                                         help="Percentile threshold for anomaly detection")
        
        with col3:
            max_gap = st.slider("Max Merge Gap (samples)", min_value=100, max_value=3000, value=1500,
                              help="Maximum gap between anomalies to consider them part of the same event")
        
        # Run analysis button
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Running anomaly detection..."):
                # Calculate sample time for merging intervals
                sample_time = None
                if len(st.session_state.data) > 1:
                    sample_time = st.session_state.data['Time'].iloc[1] - st.session_state.data['Time'].iloc[0]
                
                # Run anomaly detection
                anomalies, mse, anomaly_times, original_intervals, threshold = detect_anomalies(
                    st.session_state.data,
                    st.session_state.detector_model,
                    st.session_state.detector_scaler,
                    sequence_length,
                    anomaly_percentile
                )
                
                # Merge intervals
                merged_intervals = merge_intervals(original_intervals, max_gap, sample_time)
                
                # Save results to session state
                st.session_state.anomalies = anomalies
                st.session_state.mse_scores = mse
                st.session_state.anomaly_times = anomaly_times
                st.session_state.merged_intervals = merged_intervals
                st.session_state.threshold = threshold
                
            with st.spinner("Classifying anomalies..."):
                # Run classification
                if merged_intervals:
                    predictions = predict_anomaly_intervals(
                        st.session_state.data,
                        merged_intervals,
                        st.session_state.classifier_model,
                        st.session_state.classifier_scaler,
                        st.session_state.class_names
                    )
                    
                    # Format predictions
                    formatted_predictions = format_predictions(predictions)
                    
                    # Save to session state
                    st.session_state.predictions = formatted_predictions
                    
                    st.success(f"✅ Analysis complete! Found {len(merged_intervals)} anomalous events.")
                else:
                    st.info("No anomalies detected in the data.")
                    st.session_state.predictions = pd.DataFrame()

# Tab 3: Analysis Results
with tabs[2]:
    st.header("Step 3: Analysis Results")
    
    if st.session_state.predictions is not None:
        if len(st.session_state.predictions) > 0:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Total Anomalies</div>
                </div>
                """.format(len(st.session_state.merged_intervals)), unsafe_allow_html=True)
            
            with col2:
                if st.session_state.predictions is not None and 'predicted_fault' in st.session_state.predictions.columns:
                    unique_faults = st.session_state.predictions['predicted_fault'].nunique()
                else:
                    unique_faults = 0
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Unique Fault Types</div>
                </div>
                """.format(unique_faults), unsafe_allow_html=True)
            
            with col3:
                if st.session_state.predictions is not None and 'duration' in st.session_state.predictions.columns:
                    avg_duration = st.session_state.predictions['duration'].mean()
                    avg_duration_str = f"{avg_duration:.3f}s"
                else:
                    avg_duration_str = "N/A"
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Avg. Anomaly Duration</div>
                </div>
                """.format(avg_duration_str), unsafe_allow_html=True)
            
            # Display predictions
            st.subheader("Detected Anomalies")
            st.dataframe(st.session_state.predictions, use_container_width=True)
            
            # Fault type distribution
            st.subheader("Fault Type Distribution")
            
            if 'predicted_fault' in st.session_state.predictions.columns:
                fault_counts = st.session_state.predictions['predicted_fault'].value_counts().reset_index()
                fault_counts.columns = ['Fault Type', 'Count']
                
                fig = px.bar(fault_counts, x='Fault Type', y='Count', 
                           title='Distribution of Detected Fault Types',
                           color='Fault Type',
                           text='Count')
                
                fig.update_layout(
                    xaxis_title='Fault Type',
                    yaxis_title='Count',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly timeline
            st.subheader("Anomaly Timeline")
            
            if st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
                fig = px.timeline(
                    st.session_state.predictions, 
                    x_start='start_time', 
                    x_end='end_time', 
                    y='predicted_fault',
                    color='predicted_fault',
                    title='Anomaly Timeline',
                    labels={
                        'predicted_fault': 'Fault Type',
                        'start_time': 'Start Time',
                        'end_time': 'End Time'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Fault Type',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies were detected in the data.")
    else:
        st.info("Please run the analysis first to see results.")

# Tab 4: Visualization
with tabs[3]:
    st.header("Step 4: Detailed Visualization")
    
    if st.session_state.data is not None and st.session_state.merged_intervals is not None:
        # Select visualization type
        viz_type = st.radio(
            "Select visualization type:",
            ["Signal Visualization with Anomalies", "Reconstruction Error", "Feature Importance"],
            horizontal=True
        )
        
        if viz_type == "Signal Visualization with Anomalies":
            # Select signals to visualize
            signals = st.multiselect(
                "Select signals to visualize:", 
                ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic'],
                default=['Va', 'Vb', 'Vc']
            )
            
            if signals:
                # Create subplots for each signal
                fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, 
                                  subplot_titles=signals, vertical_spacing=0.05)
                
                # Sample data for faster plotting if dataset is large
                sample_data = st.session_state.data
                if len(sample_data) > 10000:
                    sample_step = max(1, len(sample_data) // 10000)
                    sample_data = sample_data.iloc[::sample_step].reset_index(drop=True)
                
                # Add each signal to its subplot
                for i, signal in enumerate(signals):
                    fig.add_trace(
                        go.Scatter(
                            x=sample_data['Time'], 
                            y=sample_data[signal],
                            mode='lines',
                            name=signal,
                            line=dict(width=1)
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add shapes for anomaly regions
                    for start, end in st.session_state.merged_intervals:
                        fig.add_shape(
                            type="rect",
                            x0=start, x1=end,
                            y0=sample_data[signal].min(), y1=sample_data[signal].max(),
                            fillcolor="red",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                            row=i+1, col=1
                        )
                
                # Update layout
                fig.update_layout(
                    height=150 * len(signals),
                    title_text="Signals with Detected Anomalies",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                fig.update_xaxes(title_text="Time", row=len(signals), col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                if st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
                    st.subheader("Annotation Legend")
                    for index, row in st.session_state.predictions.iterrows():
                        st.markdown(f"**Time {row['start_time']:.3f}s to {row['end_time']:.3f}s:** {row['predicted_fault']} (Confidence: {row['confidence']})")
        
        elif viz_type == "Reconstruction Error":
            if st.session_state.mse_scores is not None and st.session_state.anomaly_times is not None:
                # Plot reconstruction error
                fig = go.Figure()
                
                # Add reconstruction error line
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.anomaly_times,
                        y=st.session_state.mse_scores,
                        mode='lines',
                        name='Reconstruction Error',
                        line=dict(color='blue', width=1)
                    )
                )
                
                # Add threshold line
                fig.add_trace(
                    go.Scatter(
                        x=[st.session_state.anomaly_times.min(), st.session_state.anomaly_times.max()],
                        y=[st.session_state.threshold, st.session_state.threshold],
                        mode='lines',
                        name='Threshold',
                        line=dict(color='red', width=2, dash='dash')
                    )
                )
                
                # Add anomaly points
                if hasattr(st.session_state, 'anomalies') and st.session_state.anomalies is not None:
                    anomaly_indices = np.where(st.session_state.anomalies)[0]
                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state.anomaly_times[anomaly_indices],
                            y=st.session_state.mse_scores[anomaly_indices],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=8)
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title='Reconstruction Error with Anomaly Detection',
                    xaxis_title='Time',
                    yaxis_title='Mean Squared Error',
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display histogram of reconstruction error
                fig_hist = px.histogram(
                    st.session_state.mse_scores, 
                    nbins=50,
                    title='Distribution of Reconstruction Errors',
                    labels={'value': 'Reconstruction Error', 'count': 'Frequency'},
                    marginal='box'
                )
                
                fig_hist.add_vline(
                    x=st.session_state.threshold,
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Threshold",
                    annotation_position="top right"
                )
                
                fig_hist.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
        
        elif viz_type == "Feature Importance":
            if (st.session_state.predictions is not None and 
                len(st.session_state.predictions) > 0 and
                hasattr(st.session_state.classifier_model, 'feature_importances_')):
                
                # Get feature names
                if st.session_state.merged_intervals:
                    features = extract_features(st.session_state.data, st.session_state.merged_intervals[0])
                    feature_names = list(features.keys())
                    
                    # Get feature importances
                    importances = st.session_state.classifier_model.feature_importances_
                    
                    # Make sure we have the right number of feature names
                    if len(feature_names) == len(importances):
                        # Create DataFrame for plotting
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importances
                        fig = px.bar(
                            importance_df.head(20),  # Top 20 features
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 20 Feature Importances',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(
                            height=600,
                            yaxis={'categoryorder': 'total ascending'},
                            margin=dict(l=10, r=10, t=50, b=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature groups
                        st.subheader("Feature Importance by Signal Type")
                        
                        # Group features by signal type
                        signal_importances = {}
                        for feature, importance in zip(feature_names, importances):
                            for signal in ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']:
                                if signal in feature:
                                    if signal not in signal_importances:
                                        signal_importances[signal] = 0
                                    signal_importances[signal] += importance
                        
                        # Create chart
                        signal_df = pd.DataFrame({
                            'Signal': list(signal_importances.keys()),
                            'Importance': list(signal_importances.values())
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.pie(
                            signal_df,
                            values='Importance',
                            names='Signal',
                            title='Importance by Signal Type',
                            color='Signal',
                            hole=0.4
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Feature names and importance scores don't match in length.")
            else:
                if not hasattr(st.session_state.classifier_model, 'feature_importances_'):
                    st.info("The classifier model doesn't provide feature importance information.")
                else:
                    st.info("Run the analysis first to see feature importance.")
    else:
        st.info("Please import data and run the analysis first.")

# Add export functionality
st.sidebar.title("⚙️ Export Options")

if st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
    # Create CSV download button
    csv = st.session_state.predictions.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv",
    )
    
    # Summary report
    st.sidebar.subheader("Summary Report")
    
    # Generate summary info
    if 'predicted_fault' in st.session_state.predictions.columns:
        fault_counts = st.session_state.predictions['predicted_fault'].value_counts()
        fault_summary = "\n".join([f"- {fault}: {count}" for fault, count in fault_counts.items()])
        
        summary_text = f"""# Power System Anomaly Detection Report
Date: {time.strftime('%Y-%m-%d')}

## Summary Statistics
- Total anomalies detected: {len(st.session_state.predictions)}
- Unique fault types: {st.session_state.predictions['predicted_fault'].nunique()}
- Average anomaly duration: {st.session_state.predictions['duration'].mean():.3f}s

## Fault Type Distribution
{fault_summary}
"""
        
        st.sidebar.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name="anomaly_detection_summary.md",
            mime="text/markdown",
        )

# Help information in sidebar
st.sidebar.subheader("ℹ️ Help")
with st.sidebar.expander("How to use this dashboard"):
    st.markdown("""
    1. **Data Import**: Upload your CSV file with power system data
    2. **Model Import**: Upload your pre-trained models:
       - Detector model (.h5) and scaler for anomaly detection
       - Classifier model, scaler, and class names for fault classification
    3. **Analysis**: Run the analysis with customized parameters
    4. **Results**: View detailed analysis results and visualizations
    """)

with st.sidebar.expander("About the Models"):
    st.markdown("""
    This dashboard uses:
    - **Anomaly Detector**: LSTM-based autoencoder that detects anomalies by reconstructing normal patterns
    - **Fault Classifier**: Machine learning classifier that categorizes detected anomalies into specific fault types
    
    The detection process:
    1. Data is preprocessed and fed through the autoencoder
    2. High reconstruction errors indicate anomalies
    3. Anomalies are merged into events
    4. Features are extracted and fed to the classifier
    5. The classifier identifies the fault type
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("⚡ Power System Anomaly Dashboard")
st.sidebar.markdown("Created with Streamlit")