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

   python app3.py

Alternatively, you can use:

.. code-block:: bash

   streamlit run app3.py

The dashboard will start and be accessible at http://localhost:8501 in your web browser.

.. note::
   The main dashboard application is contained in `app3.py`, which represents the latest version of the PowerAI interface with enhanced features and improved user experience.

**Command-line Options:**

You can customize the dashboard launch by passing Streamlit arguments:

.. code-block:: bash

   streamlit run app3.py --port 8502       # Use a different port
   streamlit run app3.py --theme.base dark # Use dark theme
   streamlit run app3.py --server.headless true # Run in headless mode

Dashboard Architecture
----------------------

The `app3.py` application is built with a modular architecture:

- **Core Dashboard**: Main interface for data upload, analysis, and visualization
- **Model Integration**: Seamless integration with LSTM autoencoders and transformer-based classifiers
- **Advanced Visualization**: Interactive plots using Plotly for enhanced user experience
- **Export Capabilities**: Comprehensive reporting in CSV and PDF formats
- **Smart Query Interface**: Natural language processing capabilities (covered in the next section)

The dashboard imports the NLP query interface from `nlp_query_interface.py`, which provides intelligent data querying capabilities that will be detailed in the following section.

Dashboard Interface
-------------------

The dashboard is organized into four main tabs:

1. **📊 Data Upload & Visualization**: Upload and preview power system data
2. **⚙️ Analysis Configuration**: Configure detection parameters and run analysis
3. **📈 Results & Export**: View results and export reports
4. **🤖 Smart Query**: Natural language interface for data exploration

Data Upload & Visualization Tab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/data_upload_tab.png
   :alt: Data Upload & Visualization Tab
   :align: center

This tab provides comprehensive data management capabilities:

**Data Upload Features:**
- **CSV File Upload**: Drag-and-drop or browse for CSV files
- **Real-time Validation**: Automatic validation of required columns
- **Data Preview**: Interactive preview of uploaded data with statistics
- **Quality Metrics**: Display of data quality indicators

**Visualization Features:**
- **Signal Plots**: Interactive voltage and current signal visualization
- **Multi-signal Selection**: Choose specific signals to display
- **Responsive Design**: Plots adapt to different screen sizes
- **Export Options**: Save visualizations as images

Data Format Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

Your CSV file should follow this format:

.. code-block:: none

   Time,Va,Vb,Vc,Ia,Ib,Ic
   0.000,220.1,220.3,219.8,5.1,5.2,5.0
   0.001,220.2,220.1,219.9,5.2,5.1,5.1
   ...

**Column Specifications:**
- `Time`: Timestamp in seconds (numeric)
- `Va`, `Vb`, `Vc`: Three-phase voltage signals (numeric)
- `Ia`, `Ib`, `Ic`: Three-phase current signals (numeric)

**Data Quality Requirements:**
- Regular sampling rate recommended
- No missing values in critical columns
- Minimum 1000 samples for reliable analysis
- Consistent units across measurements

Analysis Configuration Tab
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/analysis_config_tab.png
   :alt: Analysis Configuration Tab
   :align: center

This tab provides sophisticated analysis configuration:

**Detection Parameters:**
- **Sequence Length**: Window size for LSTM autoencoder (20-200 samples)
- **Threshold Percentile**: Sensitivity control for anomaly detection (90-99%)
- **Advanced Thresholds**: Dynamic threshold adjustment based on data characteristics

**Post-processing Parameters:**
- **Maximum Gap for Merging**: Controls interval merging (100-3000 samples)
- **Classification Sequence Length**: Window size for fault classification (64-256 samples)
- **Filtering Options**: Additional noise reduction and signal conditioning

**Model Integration:**
The tab automatically loads pre-trained models:
- LSTM Autoencoder for anomaly detection (`detector/` directory)
- Transformer-based classifier for fault classification (`classifier_transformer/` directory)

Analysis Workflow
^^^^^^^^^^^^^^^^^

1. **Data Preprocessing**: Automatic scaling and sequence generation
2. **Anomaly Detection**: LSTM autoencoder reconstruction error analysis
3. **Threshold Calculation**: Dynamic threshold based on percentile selection
4. **Interval Merging**: Consolidation of nearby anomalies into fault intervals
5. **Fault Classification**: Multi-class classification of detected intervals
6. **Results Compilation**: Comprehensive results with confidence metrics

Results & Export Tab
~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/results_tab.png
   :alt: Results & Export Tab
   :align: center

This tab presents comprehensive analysis results:

**Summary Metrics:**
- **Detection Statistics**: Total anomalies, detection rate, fault intervals
- **Classification Results**: Successful classifications and confidence levels
- **Performance Metrics**: Analysis duration and processing statistics

**Interactive Visualizations:**
- **Reconstruction Error Plot**: Time series of reconstruction errors with threshold
- **Signal Overlays**: Original signals with highlighted fault regions
- **Classification Confidence**: Distribution of classification confidence scores
- **Fault Type Distribution**: Pie charts and histograms of detected fault types

**Detailed Results Table:**
- **Fault Intervals**: Start/end times, duration, and classification results
- **Confidence Scores**: Classification confidence with progress bars
- **Sortable Columns**: Interactive sorting and filtering capabilities

Export Capabilities
^^^^^^^^^^^^^^^^^^^

**CSV Export:**
- Comprehensive data export with summary statistics
- Fault interval details with timestamps and classifications
- Analysis parameters and model configuration

**PDF Reports:**
- Professional reports with charts and tables
- Summary statistics and detailed findings
- Configurable report templates with company branding

Smart Query Tab
~~~~~~~~~~~~~~~

The Smart Query tab provides an intelligent interface for data exploration using natural language processing capabilities. This advanced feature allows users to interact with their power system data using conversational queries.

.. note::
   Detailed documentation for the Smart Query interface, including its natural language processing capabilities and query examples, is provided in the next section of this documentation.

Key Features of app3.py
------------------------

Enhanced User Experience
~~~~~~~~~~~~~~~~~~~~~~~~

The `app3.py` implementation includes several UX improvements:

- **Modern Styling**: Custom CSS with gradient backgrounds and modern design elements
- **Responsive Layout**: Adaptive interface that works on different screen sizes
- **Interactive Elements**: Hover effects, progress indicators, and real-time feedback
- **Error Handling**: Comprehensive error messages and recovery suggestions

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Caching**: Strategic use of Streamlit's caching for model loading and data processing
- **Memory Management**: Efficient handling of large datasets
- **Progressive Loading**: Staged loading of components for faster initial response
- **Session Management**: Persistent session state across tab navigation

Model Integration Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The application seamlessly integrates multiple AI models:

**LSTM Autoencoder Integration:**
- Automatic model loading with error handling
- Preprocessing pipeline with saved scalers
- Configurable sequence generation
- Real-time reconstruction error calculation

**Transformer Classifier Integration:**
- Multi-input model support (time series + statistical features)
- Statistical feature extraction pipeline
- Confidence score calculation and visualization
- Class label mapping and interpretation

Advanced Visualization Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Interactive Plotly Charts:**
- Zoomable and pannable time series plots
- Multi-subplot layouts for signal comparison
- Customizable color schemes and styling
- Export capabilities for presentations

**Real-time Updates:**
- Dynamic plot updates based on user selections
- Responsive threshold visualization
- Interactive anomaly highlighting
- Progressive result rendering

Production Deployment
--------------------

Deploying app3.py in Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production deployment, consider:

**Server Configuration:**
- Use production-grade WSGI servers (Gunicorn, uWSGI)
- Configure reverse proxy (Nginx, Apache)
- Set up SSL/TLS certificates for secure access
- Implement load balancing for high availability

**Security Considerations:**
- User authentication and authorization
- Data encryption at rest and in transit
- Input validation and sanitization
- Audit logging for compliance

**Scalability Options:**
- Containerization with Docker
- Kubernetes orchestration for cloud deployment
- Auto-scaling based on usage patterns
- Database integration for persistent storage

**Monitoring and Maintenance:**
- Application performance monitoring
- Error tracking and alerting
- Regular model updates and retraining
- Backup and disaster recovery procedures

Customization Guide
-------------------

Extending app3.py
~~~~~~~~~~~~~~~~~

The modular architecture allows for easy customization:

**Adding New Visualizations:**

.. code-block:: python

   # Add custom visualization functions
   def create_custom_plot(data, results):
       # Your custom plotting logic
       return fig

**Integrating Additional Models:**

.. code-block:: python

   # Add new model types
   @st.cache_resource
   def load_custom_model():
       # Model loading logic
       return model

**Custom Export Formats:**

.. code-block:: python

   # Add new export options
   def generate_custom_report(results):
       # Report generation logic
       return report_data

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

Key configuration parameters in `app3.py`:

- **Model Paths**: Directory locations for saved models
- **UI Themes**: Color schemes and styling options
- **Performance Settings**: Caching strategies and memory limits
- **Export Options**: Available formats and templates

Troubleshooting
---------------

Common Issues with app3.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Startup Issues:**
- Verify all required dependencies are installed
- Check model file availability and permissions
- Ensure proper Python environment activation

**Performance Issues:**
- Monitor memory usage with large datasets
- Adjust caching parameters for optimization
- Consider data sampling for initial exploration

**Model Loading Errors:**
- Verify model file compatibility
- Check TensorFlow/Keras version compatibility
- Ensure all required model files are present

**Visualization Problems:**
- Clear browser cache if plots don't render
- Check JavaScript console for errors
- Verify Plotly.js compatibility

Best Practices
--------------

Using app3.py Effectively
~~~~~~~~~~~~~~~~~~~~~~~~

**Data Preparation:**
- Preprocess data to ensure quality
- Use consistent sampling rates
- Remove or interpolate missing values
- Validate signal ranges and units

**Analysis Workflow:**
- Start with default parameters
- Iteratively adjust thresholds based on results
- Validate classifications with domain knowledge
- Document analysis parameters for reproducibility

**Result Interpretation:**
- Consider confidence scores in decision-making
- Cross-validate results with historical data
- Use multiple visualization perspectives
- Maintain analysis logs for future reference

Next Steps
----------

Now that you understand the main dashboard interface, the next section will cover the Smart Query capabilities that leverage natural language processing to provide an intuitive way to explore and analyze your power system data.

Additional Resources:
- :doc:`usage` - Practical application examples
- :doc:`api` - Programmatic access to PowerAI
- :doc:`models` - Understanding the underlying AI models
- :doc:`nlp_interface` - Smart Query interface documentation (next section)