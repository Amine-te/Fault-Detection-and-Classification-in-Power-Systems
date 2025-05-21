===================================
PowerAI: Power System Fault Detection
===================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
   :alt: Python Versions

PowerAI is an advanced AI solution for detecting and classifying faults in electrical power systems using machine learning and deep learning techniques.

Key Features
============

- Real-time anomaly detection in power system signals using autoencoder neural networks
- Automatic fault classification of detected anomalies
- Interactive dashboard for visualizing data and detected faults
- Feature importance analysis to understand fault signatures
- Comprehensive signal processing for voltage and current measurements

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   introduction
   installation
   models
   dashboard
   usage

.. toctree::
   :maxdepth: 2
   :caption: Technical Reference

   api
   contributing
   changelog

Get Started
===========

To get started with PowerAI, follow the :doc:`installation guide <installation>` and check out the :doc:`usage examples <usage>`.

System Requirements
===================

- Python 3.8+
- TensorFlow 2.6+
- Scikit-learn 1.0+
- 4GB+ RAM for model training
- GPU recommended for large datasets and complex models

License
=======

This project is licensed under the MIT License - see the LICENSE file for details.
