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
          # 🧠 NeuroSense — EEG-Based Brain Tumor Classification

**NeuroSense** is a machine learning project that explores EEG signal analysis for classifying brain-related conditions using deep learning. It focuses on preprocessing EEG data and applying a Convolutional Neural Network (CNN) to learn patterns from signal features.

The project demonstrates an end-to-end machine learning workflow, from data preparation and feature transformation to model training and evaluation.

## ✨ Features

- **EEG Data Processing:** Loads and prepares EEG data for model development.
- **Data Preprocessing:** Applies normalization and dimensional transformation to prepare inputs for learning.
- **Deep Learning:** Uses a CNN-based architecture to learn patterns from EEG signal data.
- **Binary Classification:** Groups selected dataset classes into tumor-related and normal categories according to the project’s label-mapping strategy.
- **Model Evaluation:** Evaluates classification performance using appropriate metrics.
- **End-to-End Workflow:** Brings together data preparation, model training, and evaluation in a structured pipeline.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Development | Jupyter Notebook, Google Colab |

## ⚙️ Project Workflow

1. **Data Loading:** Load the EEG dataset from a CSV file.
2. **Label Preparation:** Map the original dataset labels into the target classification categories.
3. **Preprocessing:** Prepare the input features and normalize the data.
4. **Data Splitting:** Divide the dataset into training and testing sets.
5. **Model Training:** Train a CNN to learn patterns in the processed EEG features.
6. **Evaluation:** Assess the trained model on held-out data to understand its classification performance.

## 📂 Dataset

The project uses EEG data based on the [Epileptic Seizure Recognition dataset](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition) hosted on Kaggle.

The dataset contains EEG signal measurements represented as numerical features. The original class labels must be interpreted carefully when defining the target categories for the binary classification task.

**Note:** The dataset's original labels describe seizure-related categories, so the mapping used in this project should be documented accurately. This dataset alone does not establish clinical validity for brain tumor detection.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or a compatible version
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/akanksha303/NeuroSense.git
cd NeuroSense
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scikit-learn tensorflow
```

### 4. Prepare the dataset

Download the dataset from Kaggle and place the CSV file in the location expected by the preprocessing code. Update the dataset path if necessary.

### 5. Run the project

Open the project's notebook or Python entry point and execute the preprocessing, training, and evaluation steps in sequence.

Refer to the repository files for the exact execution instructions and expected input paths.

## 📊 Results

The model's performance should be reported using the metrics obtained from the actual evaluation run.

Suggested metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

*Add your verified evaluation results here once confirmed.*

## 🔮 Future Improvements

- Improve preprocessing and model evaluation.
- Compare CNN performance with traditional machine learning models.
- Evaluate generalization on independent EEG datasets.
- Add visualizations for model performance and classification errors.
- Develop a user interface for exploring model predictions.

## ⚠️ Disclaimer

NeuroSense is an educational and research project. Its predictions are not a medical diagnosis, and the model has not been established as a clinically validated brain tumor detection system. Clinical use would require appropriate datasets, independent validation, and expert medical assessment.

## 👩‍💻 Author

**Akanksha Tripathi**

- GitHub: [@akanksha303](https://github.com/akanksha303)
- LinkedIn: [akanksha303](https://linkedin.com/in/akanksha303)

---

If you find this project interesting, consider giving the repository a ⭐ on GitHub!
