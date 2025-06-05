import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from joblib import load
import tensorflow as tf
from scipy.stats import skew, kurtosis
import warnings
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# Import the nlp query interface
from nlp_query_interface import create_tab4_interface

# Page configuration
st.set_page_config(
    page_title="⚡ Power System Fault Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 10px;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data' not in st.session_state:
    st.session_state.data = None
# if 'query_interface' not in st.session_state:
#     st.session_state.query_interface = QueryInterface()
# if 'nlp_filters' not in st.session_state:
#     st.session_state.nlp_filters = {}
# if 'nlp_action' not in st.session_state:
#     st.session_state.nlp_action = None
# if 'show_nlp_results' not in st.session_state:
#     st.session_state.show_nlp_results = False


# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>⚡ Power System Fault Detection & Classification</h1>
    <p style='font-size: 1.2rem; color: #666;'>Advanced AI-powered fault detection using LSTM Autoencoders</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for tips and information
with st.sidebar:
    st.markdown("## 📋 User Guide")
    
    with st.expander("🔍 How to Use"):
        st.markdown("""
        **Step 1:** Upload your power system data (CSV format)
        - Required columns: Time, Va, Vb, Vc, Ia, Ib, Ic
        
        **Step 2:** Configure analysis parameters
        - Adjust sequence length and thresholds
        
        **Step 3:** Run the analysis
        - View detection results and classifications
        
        **Step 4:** Export your report
        - Save as CSV or PDF format
        """)
    
    with st.expander("📊 Data Requirements"):
        st.markdown("""
        **Required Columns:**
        - `Time`: Timestamp or time index
        - `Va`, `Vb`, `Vc`: Three-phase voltages
        - `Ia`, `Ib`, `Ic`: Three-phase currents
        
        **Data Quality:**
        - Uniform sampling rate recommended
        - No missing values in critical columns
        - Sufficient data length (>1000 samples)
        """)
    
    with st.expander("⚙️ Model Information"):
        st.markdown("""
        **Detection Model:**
        - LSTM Autoencoder for anomaly detection
        - Reconstruction error threshold-based detection
        
        **Classification Model:**
        - Transformer for classification
        - Statistical feature extraction
        - Multi-class fault type identification
        """)

# Function to load models
@st.cache_resource
def load_models():
    try:
        # Load detector components
        detector_scaler = load('detector/scaler.save')
        detector_model = tf.keras.models.load_model(
            'detector/lstm_autoencoder_anomaly_detection.h5',
            custom_objects={'mse': 'mse'}
        )
        
        # Load classifier components
        classifier_model = tf.keras.models.load_model('classifier_transformer/best_fault_classifier.h5')
        classifier_package = load('classifier_transformer/complete_dl_fault_system.joblib')
        
        return {
            'detector_scaler': detector_scaler,
            'detector_model': detector_model,
            'classifier_model': classifier_model,
            'classifier_package': classifier_package
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure model files are in the correct directories:\n- detector/\n- classifier_transformer/")
        return None

# Function definitions
def preprocess_new_data(new_data, scaler, sequence_length):
    scaled_data = scaler.transform(new_data[['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']])
    X_new = []
    for i in range(len(scaled_data) - sequence_length + 1):
        X_new.append(scaled_data[i:i+sequence_length])
    return np.array(X_new)

def merge_intervals(intervals, max_gap_samples=500, sample_time=0.001):
    if not intervals:
        return []
    
    max_gap_time = max_gap_samples * sample_time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [list(sorted_intervals[0])]
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        if current_start <= last_end + max_gap_time:
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])
    
    return [tuple(interval) for interval in merged]

def extract_statistical_features(window, signal_columns):
    features = []
    
    for signal in signal_columns:
        if signal in window.columns:
            sig_data = window[signal].values
            if len(sig_data) > 1:
                features.extend([
                    np.mean(sig_data),
                    np.std(sig_data),
                    skew(sig_data),
                    kurtosis(sig_data),
                    np.sqrt(np.mean(sig_data ** 2)),  # RMS
                    np.ptp(sig_data),  # Peak-to-peak
                    np.max(np.abs(sig_data)) / (np.sqrt(np.mean(sig_data ** 2)) + 1e-9)  # Crest factor
                ])
    
    # Cross-signal correlations
    try:
        if len(window) > 1 and len(signal_columns) >= 2:
            for i in range(len(signal_columns)-1):
                for j in range(i+1, len(signal_columns)):
                    if signal_columns[i] in window.columns and signal_columns[j] in window.columns:
                        corr = np.corrcoef(window[signal_columns[i]], window[signal_columns[j]])[0,1]
                        features.append(corr if not np.isnan(corr) else 0)
    except:
        pass
    
    return np.array(features)

def classify_fault_interval(interval_start, interval_end, data, models, sequence_length=128):
    start_idx = np.searchsorted(data['Time'], interval_start)
    end_idx = np.searchsorted(data['Time'], interval_end)
    window = data.iloc[start_idx:end_idx]
    
    if len(window) < sequence_length:
        return None, None, "Insufficient data"
    
    signal_columns = ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
    available_signals = [col for col in signal_columns if col in window.columns]
    
    if not available_signals:
        return None, None, "No signal data"
    
    # Create time series sequences
    time_series_data = window[available_signals].values
    
    if len(time_series_data) >= sequence_length:
        mid_start = (len(time_series_data) - sequence_length) // 2
        sequence = time_series_data[mid_start:mid_start + sequence_length]
        
        # Scale the sequence
        seq_scaler = models['classifier_package']['sequence_scaler']
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_scaled = seq_scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(sequence.shape)
        
        # Extract statistical features
        feat_scaler = models['classifier_package']['feature_scaler']
        features = extract_statistical_features(window, available_signals)
        features_scaled = feat_scaler.transform(features.reshape(1, -1))
        
        # Prepare inputs for the model
        X_seq = sequence_scaled.reshape(1, sequence_length, len(available_signals))
        X_feat = features_scaled
        
        # Make prediction
        predictions = models['classifier_model'].predict([X_seq, X_feat], verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        label_encoder = models['classifier_package']['label_encoder']
        predicted_fault_type = label_encoder.classes_[predicted_class_idx]
        
        return predicted_fault_type, confidence, "Success"
    
    return None, None, "Processing error"

def generate_pdf_report(results, data_info):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Power System Fault Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Analysis Summary
    story.append(Paragraph("Analysis Summary", styles['Heading2']))
    summary_data = [
        ['Parameter', 'Value'],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Data Points', str(data_info['total_samples'])],
        ['Analysis Duration', f"{data_info['duration']:.2f} seconds"],
        ['Detected Anomalies', str(results['anomaly_count'])],
        ['Detection Rate', f"{results['anomaly_rate']:.1f}%"],
        ['Merged Intervals', str(len(results['merged_intervals']))]
    ]
    
    table = Table(summary_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Classification Results
    if results['classification_results']:
        story.append(Paragraph("Fault Classification Results", styles['Heading2']))
        class_data = [['Interval', 'Start Time', 'End Time', 'Fault Type', 'Confidence']]
        
        for i, result in enumerate(results['classification_results']):
            if result['fault_type']:
                class_data.append([
                    str(i+1),
                    f"{result['start_time']:.3f}",
                    f"{result['end_time']:.3f}",
                    result['fault_type'],
                    f"{result['confidence']:.3f}"
                ])
        
        if len(class_data) > 1:
            class_table = Table(class_data)
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(class_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main application tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Upload & Visualization", "⚙️ Analysis Configuration", "📈 Results & Export", "🤖 Smart Query"])

with tab1:
    st.markdown("## 📊 Data Upload & Visualization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your power system data (CSV format)",
            type=['csv'],
            help="File should contain columns: Time, Va, Vb, Vc, Ia, Ib, Ic"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                st.markdown('<div class="success-box">✅ Data uploaded successfully!</div>', unsafe_allow_html=True)
                
                # Data preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Data statistics
                st.markdown("### 📊 Data Statistics")
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    st.markdown(f'<div class="metric-container"><h3>{len(data)}</h3><p>Total Samples</p></div>', unsafe_allow_html=True)
                
                with col_stats2:
                    st.markdown(f'<div class="metric-container"><h3>{len(data.columns)}</h3><p>Columns</p></div>', unsafe_allow_html=True)
                
                with col_stats3:
                    missing_vals = data.isnull().sum().sum()
                    st.markdown(f'<div class="metric-container"><h3>{missing_vals}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
                
                with col_stats4:
                    duration = data['Time'].iloc[-1] - data['Time'].iloc[0] if 'Time' in data.columns else 0
                    st.markdown(f'<div class="metric-container"><h3>{duration:.2f}s</h3><p>Duration</p></div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.markdown('<div class="info-box" style="color: #222;"><h4>💡 Tips for Data Upload</h4><ul><li>Ensure your CSV has proper headers</li><li>Time column should be numeric</li><li>Check for missing values</li><li>Verify signal units are consistent</li></ul></div>', unsafe_allow_html=True)
    
    # Signal visualization
    if st.session_state.data is not None:
        st.markdown("### 📈 Signal Visualization")
        
        data = st.session_state.data
        required_cols = ['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic']
        available_cols = [col for col in required_cols if col in data.columns]
        
        if available_cols:
            # Voltage signals
            st.markdown("#### ⚡ Voltage Signals")
            voltage_cols = [col for col in ['Va', 'Vb', 'Vc'] if col in available_cols]
            
            if voltage_cols:
                selected_v_signals = st.multiselect(
                    "Select voltage signals to display:",
                    voltage_cols,
                    default=voltage_cols,
                    key="voltage_select"
                )
                
                if selected_v_signals:
                    fig_v = make_subplots(
                        rows=len(selected_v_signals), cols=1,
                        subplot_titles=selected_v_signals,
                        shared_xaxes=True,
                        vertical_spacing=0.1
                    )
                    
                    colors_v = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    for i, signal in enumerate(selected_v_signals):
                        fig_v.add_trace(
                            go.Scatter(
                                x=data['Time'],
                                y=data[signal],
                                name=signal,
                                line=dict(color=colors_v[i % len(colors_v)], width=1.5)
                            ),
                            row=i+1, col=1
                        )
                    
                    fig_v.update_layout(
                        height=200 * len(selected_v_signals),
                        title_text="Voltage Signals vs Time",
                        showlegend=False
                    )
                    fig_v.update_xaxes(title_text="Time", row=len(selected_v_signals), col=1)
                    
                    st.plotly_chart(fig_v, use_container_width=True)
            
            # Current signals
            st.markdown("#### ⚡ Current Signals")
            current_cols = [col for col in ['Ia', 'Ib', 'Ic'] if col in available_cols]
            
            if current_cols:
                selected_i_signals = st.multiselect(
                    "Select current signals to display:",
                    current_cols,
                    default=current_cols,
                    key="current_select"
                )
                
                if selected_i_signals:
                    fig_i = make_subplots(
                        rows=len(selected_i_signals), cols=1,
                        subplot_titles=selected_i_signals,
                        shared_xaxes=True,
                        vertical_spacing=0.1
                    )
                    
                    colors_i = ['#d62728', '#9467bd', '#8c564b']
                    for i, signal in enumerate(selected_i_signals):
                        fig_i.add_trace(
                            go.Scatter(
                                x=data['Time'],
                                y=data[signal],
                                name=signal,
                                line=dict(color=colors_i[i % len(colors_i)], width=1.5)
                            ),
                            row=i+1, col=1
                        )
                    
                    fig_i.update_layout(
                        height=200 * len(selected_i_signals),
                        title_text="Current Signals vs Time",
                        showlegend=False
                    )
                    fig_i.update_xaxes(title_text="Time", row=len(selected_i_signals), col=1)
                    
                    st.plotly_chart(fig_i, use_container_width=True)
        else:
            st.warning("⚠️ Required signal columns not found in the data.")

with tab2:
    st.markdown("## ⚙️ Analysis Configuration")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Visualization' tab.")
    else:
        # Load models
        models = load_models()
        
        if models is not None:
            st.markdown('<div class="success-box">✅ Models loaded successfully!</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🔧 Detection Parameters")
                
                sequence_length = st.slider(
                    "Sequence Length (for anomaly detection)",
                    min_value=20,
                    max_value=200,
                    value=50,
                    help="Number of consecutive samples used for anomaly detection"
                )
                
                threshold_percentile = st.slider(
                    "Threshold Percentile",
                    min_value=90,
                    max_value=99,
                    value=95,
                    help="Percentile used to determine anomaly threshold"
                )
                
                st.markdown("### 🔄 Post-processing Parameters")
                
                max_gap_samples = st.slider(
                    "Maximum Gap for Merging (samples)",
                    min_value=100,
                    max_value=3000,
                    value=1500,
                    help="Maximum gap between anomalies to merge them into a single fault interval"
                )
                
                classification_sequence_length = st.slider(
                    "Classification Sequence Length",
                    min_value=64,
                    max_value=256,
                    value=128,
                    help="Sequence length for fault classification"
                )
                
                # Run analysis button
                if st.button("🚀 Run Fault Detection & Classification", type="primary", use_container_width=True):
                    with st.spinner("Running analysis... This may take a few minutes."):
                        try:
                            data = st.session_state.data
                            
                            # Preprocessing
                            X_sequences = preprocess_new_data(data, models['detector_scaler'], sequence_length)
                            
                            # Anomaly detection
                            reconstructions = models['detector_model'].predict(X_sequences, verbose=0)
                            mse = np.mean(np.square(X_sequences - reconstructions), axis=(1,2))
                            
                            # Dynamic threshold
                            threshold = np.percentile(mse, threshold_percentile)
                            anomalies = mse > threshold
                            
                            # Get timestamps
                            time_stamps = data['Time'].values[sequence_length - 1:]
                            
                            # Get anomaly intervals
                            anomaly_indices = np.where(anomalies)[0]
                            original_intervals = []
                            
                            for win_idx in anomaly_indices:
                                start_idx = win_idx
                                end_idx = win_idx + sequence_length - 1
                                start_time = data['Time'].iloc[start_idx]
                                end_time = data['Time'].iloc[end_idx]
                                original_intervals.append((start_time, end_time))
                            
                            # Merge intervals
                            sample_time = data['Time'].iloc[1] - data['Time'].iloc[0] if len(data) > 1 else 0.001
                            merged_intervals = merge_intervals(original_intervals, max_gap_samples, sample_time)
                            
                            # Classify faults
                            classification_results = []
                            for i, (start_time, end_time) in enumerate(merged_intervals):
                                fault_type, confidence, status = classify_fault_interval(
                                    start_time, end_time, data, models, classification_sequence_length
                                )
                                
                                result = {
                                    'interval': i + 1,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'duration': end_time - start_time,
                                    'fault_type': fault_type,
                                    'confidence': confidence,
                                    'status': status
                                }
                                classification_results.append(result)
                            
                            # Store results
                            results = {
                                'mse': mse,
                                'threshold': threshold,
                                'anomalies': anomalies,
                                'time_stamps': time_stamps,
                                'merged_intervals': merged_intervals,
                                'classification_results': classification_results,
                                'anomaly_count': np.sum(anomalies),
                                'anomaly_rate': (np.sum(anomalies) / len(anomalies)) * 100,
                                'parameters': {
                                    'sequence_length': sequence_length,
                                    'threshold_percentile': threshold_percentile,
                                    'max_gap_samples': max_gap_samples,
                                    'classification_sequence_length': classification_sequence_length
                                }
                            }
                            
                            st.session_state.results = results
                            st.session_state.analysis_complete = True
                            
                            st.success("✅ Analysis completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
            
            with col2:
                st.markdown('<div class="info-box" style="color: #222;"><h4>🔧 Parameter Guidelines</h4><ul><li><b>Sequence Length:</b> Longer sequences capture more temporal patterns but require more computation</li><li><b>Threshold Percentile:</b> Higher values reduce false positives but may miss subtle faults</li><li><b>Max Gap:</b> Larger gaps merge more intervals but may combine distinct faults</li></ul></div>', unsafe_allow_html=True)
        else:
            st.error("❌ Could not load required models. Please check model files.")

with tab3:
    st.markdown("## 📈 Results & Export")
    
    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first in the 'Analysis Configuration' tab.")
    else:
        results = st.session_state.results
        data = st.session_state.data
        
        # Results summary
        st.markdown("### 📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-container"><h3>{results["anomaly_count"]}</h3><p>Anomalies Detected</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-container"><h3>{results["anomaly_rate"]:.1f}%</h3><p>Detection Rate</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-container"><h3>{len(results["merged_intervals"])}</h3><p>Fault Intervals</p></div>', unsafe_allow_html=True)
        
        with col4:
            successful_classifications = len([r for r in results['classification_results'] if r['fault_type']])
            st.markdown(f'<div class="metric-container"><h3>{successful_classifications}</h3><p>Classifications</p></div>', unsafe_allow_html=True)
        
        # Reconstruction error plot
        st.markdown("### 🔍 Anomaly Detection Results")
        
        fig_mse = go.Figure()
        
        # Plot reconstruction error
        fig_mse.add_trace(go.Scatter(
            x=results['time_stamps'],
            y=results['mse'],
            mode='lines',
            name='Reconstruction Error',
            line=dict(color='steelblue', width=1.5)
        ))
        
        # Plot threshold
        fig_mse.add_hline(
            y=results['threshold'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Threshold ({results['threshold']:.4f})"
        )
        
        # Highlight anomalies
        anomaly_times = results['time_stamps'][results['anomalies']]
        anomaly_errors = results['mse'][results['anomalies']]
        
        fig_mse.add_trace(go.Scatter(
            x=anomaly_times,
            y=anomaly_errors,
            mode='markers',
            name='Detected Anomalies',
            marker=dict(color='red', size=6, symbol='circle')
        ))
        
        fig_mse.update_layout(
            title="Reconstruction Error Analysis",
            xaxis_title="Time",
            yaxis_title="Reconstruction Error",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_mse, use_container_width=True)
        
        # Signal plots with highlighted anomalies
        st.markdown("### ⚡ Signals with Detected Faults")
        
        signal_groups = [
            (['Va', 'Vb', 'Vc'], 'Voltage Signals', ['#1f77b4', '#ff7f0e', '#2ca02c']),
            (['Ia', 'Ib', 'Ic'], 'Current Signals', ['#d62728', '#9467bd', '#8c564b'])
        ]
        
        for signals, title, colors in signal_groups:
            available_signals = [s for s in signals if s in data.columns]
            if available_signals:
                fig_signals = make_subplots(
                    rows=len(available_signals), cols=1,
                    subplot_titles=available_signals,
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                for i, signal in enumerate(available_signals):
                    # Plot signal
                    fig_signals.add_trace(
                        go.Scatter(
                            x=data['Time'],
                            y=data[signal],
                            name=signal,
                            line=dict(color=colors[i], width=1.2),
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                    
                    # Highlight fault intervals
                    for start_time, end_time in results['merged_intervals']:
                        fig_signals.add_vrect(
                            x0=start_time,
                            x1=end_time,
                            fillcolor="red",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                            row=i+1, col=1
                        )
                
                fig_signals.update_layout(
                    title=title,
                    height=200 * len(available_signals),
                    showlegend=False
                )
                fig_signals.update_xaxes(title_text="Time", row=len(available_signals), col=1)
                
                st.plotly_chart(fig_signals, use_container_width=True)
        
        # Classification results table
        if results['classification_results']:
            st.markdown("### 🎯 Fault Classification Results")
            
            classification_df = pd.DataFrame(results['classification_results'])
            
            # Filter successful classifications
            successful_df = classification_df[classification_df['fault_type'].notna()].copy()
            
            if not successful_df.empty:
                # Display results table
                display_df = successful_df[['interval', 'start_time', 'end_time', 'duration', 'fault_type', 'confidence']].copy()
                display_df['start_time'] = display_df['start_time'].round(3)
                display_df['end_time'] = display_df['end_time'].round(3)
                display_df['duration'] = display_df['duration'].round(3)
                display_df['confidence'] = display_df['confidence'].round(3)
                
                st.dataframe(
                    display_df,
                    column_config={
                        "interval": "Fault #",
                        "start_time": "Start Time (s)",
                        "end_time": "End Time (s)",
                        "duration": "Duration (s)",
                        "fault_type": "Fault Type",
                        "confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            help="Classification confidence",
                            min_value=0,
                            max_value=1,
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Fault type distribution
                st.markdown("#### 📊 Fault Type Distribution")
                fault_counts = successful_df['fault_type'].value_counts()
                
                fig_pie = px.pie(
                    values=fault_counts.values,
                    names=fault_counts.index,
                    title="Distribution of Detected Fault Types"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Confidence distribution
                fig_conf = px.histogram(
                    successful_df,
                    x='confidence',
                    nbins=20,
                    title="Classification Confidence Distribution",
                    labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
                )
                fig_conf.update_layout(bargap=0.1)
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.warning("⚠️ No successful fault classifications found.")
        
        # Export section
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV Export
            if st.button("📄 Export as CSV", use_container_width=True):
                # Prepare export data
                export_data = []
                
                # Add summary information
                export_data.append({
                    'Type': 'Summary',
                    'Parameter': 'Total Samples',
                    'Value': len(data),
                    'Unit': 'samples'
                })
                export_data.append({
                    'Type': 'Summary',
                    'Parameter': 'Detected Anomalies',
                    'Value': results['anomaly_count'],
                    'Unit': 'count'
                })
                export_data.append({
                    'Type': 'Summary',
                    'Parameter': 'Detection Rate',
                    'Value': f"{results['anomaly_rate']:.1f}",
                    'Unit': '%'
                })
                export_data.append({
                    'Type': 'Summary',
                    'Parameter': 'Threshold',
                    'Value': f"{results['threshold']:.6f}",
                    'Unit': 'error'
                })
                
                # Add fault intervals
                for i, (start_time, end_time) in enumerate(results['merged_intervals']):
                    result = results['classification_results'][i] if i < len(results['classification_results']) else {}
                    
                    export_data.append({
                        'Type': 'Fault',
                        'Parameter': f'Fault_{i+1}_Start',
                        'Value': f"{start_time:.3f}",
                        'Unit': 'seconds'
                    })
                    export_data.append({
                        'Type': 'Fault',
                        'Parameter': f'Fault_{i+1}_End',
                        'Value': f"{end_time:.3f}",
                        'Unit': 'seconds'
                    })
                    export_data.append({
                        'Type': 'Fault',
                        'Parameter': f'Fault_{i+1}_Type',
                        'Value': result.get('fault_type', 'Unknown'),
                        'Unit': 'class'
                    })
                    export_data.append({
                        'Type': 'Fault',
                        'Parameter': f'Fault_{i+1}_Confidence',
                        'Value': f"{result.get('confidence', 0):.3f}" if result.get('confidence') else 'N/A',
                        'Unit': 'score'
                    })
                
                # Convert to DataFrame and download
                export_df = pd.DataFrame(export_data)
                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download CSV Report",
                    data=csv_buffer.getvalue(),
                    file_name=f"fault_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # PDF Export
            if st.button("📑 Export as PDF", use_container_width=True):
                try:
                    data_info = {
                        'total_samples': len(data),
                        'duration': data['Time'].iloc[-1] - data['Time'].iloc[0] if 'Time' in data.columns else 0
                    }
                    
                    pdf_buffer = generate_pdf_report(results, data_info)
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"fault_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
        
        # Analysis parameters used
        st.markdown("### ⚙️ Analysis Parameters Used")
        params_df = pd.DataFrame([
            {'Parameter': 'Sequence Length (Detection)', 'Value': results['parameters']['sequence_length']},
            {'Parameter': 'Threshold Percentile', 'Value': f"{results['parameters']['threshold_percentile']}%"},
            {'Parameter': 'Max Gap for Merging', 'Value': f"{results['parameters']['max_gap_samples']} samples"},
            {'Parameter': 'Classification Sequence Length', 'Value': results['parameters']['classification_sequence_length']},
            {'Parameter': 'Calculated Threshold', 'Value': f"{results['threshold']:.6f}"}
        ])
        
        st.dataframe(params_df, use_container_width=True, hide_index=True)

with tab4:
    create_tab4_interface()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>⚡ Power System Fault Detection Dashboard</p>
    <p>Built with Streamlit • Advanced AI-powered Analysis</p>
</div>
""", unsafe_allow_html=True)