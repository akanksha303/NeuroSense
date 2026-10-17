<div align="center">

# 🧠 NeuroSense
### Multimodal Clinical Intelligence Platform for Neurological Risk Assessment & Brain Tumor Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%252B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-TFLite-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Bridging functional electrical activity (EEG) and structural anatomical imaging (MRI) through an intelligent multimodal late-fusion architecture.*

</div>

---

## 🌟 Overview

Traditional diagnostic tools often evaluate neurological anomalies through isolated lenses—either reviewing electrical brain waves or structural brain scans. **NeuroSense** bridges this gap by providing a **unified multimodal diagnostic pipeline**. 

Engineered with complete operational flexibility, the platform empowers clinicians and researchers to perform rapid single-modality screenings or high-confidence multimodal fusion assessments to evaluate neurological risk and classify brain tumors.

---

## 🏗️ System Architecture

```text
                  +-----------------------------------+
                  |         NeuroSense Router         |
                  +-----------------------------------+
                       /                        \
        [ EEG Upload ] /                          \ [ MRI Upload ]
                      v                            v
        +---------------------------+    +---------------------------+
        |   EEG Preprocessing &     |    |    MRI Image Preprocessing|
        |   1D-CNN Branch           |    |    & 2D-CNN Branch        |
        |   (Temporal Wave Analysis)|    |    (Spatial Scan Analysis)|
        +---------------------------+    +---------------------------+
                      \                            /
                       \                          /
                        v                        v
                  +-----------------------------------+
                  |    Decision-Level Fusion Layer    |
                  |    (Conditional Multimodal Meta)  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    FastAPI RESTful Prediction     |
                  |    & JSON Response Engine         |
                  +-----------------------------------+
