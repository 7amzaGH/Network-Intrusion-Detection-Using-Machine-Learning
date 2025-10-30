# Network Intrusion Detection System

A machine learning-based approach to detecting network intrusions and anomalies using various classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a network intrusion detection system using machine learning algorithms to classify network traffic as either normal or anomalous. The system analyzes 41 different features of network connections to identify potential security threats.

## ✨ Features

- **Multiple ML Models**: Comparison of K-Nearest Neighbors, Logistic Regression, and Decision Tree classifiers
- **Feature Selection**: Recursive Feature Elimination (RFE) to identify the most important features
- **Hyperparameter Optimization**: Optuna-based hyperparameter tuning for optimal model performance
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, and classification reports
- **Data Preprocessing**: Label encoding, standardization, and train-test splitting

## 📊 Dataset

The dataset contains network traffic data with 42 features:
- **Training Set**: 25,192 samples
- **Test Set**: 10,798 samples
- **Classes**: Binary classification (Normal vs. Anomaly)
- **Features**: 41 network traffic characteristics including:
  - Protocol type
  - Service type
  - Connection flags
  - Byte counts
  - Error rates
  - Host-based statistics

### Class Distribution
- Normal: 13,449 samples (53.4%)
- Anomaly: 11,743 samples (46.6%)

## 🤖 Models Used

### 1. K-Nearest Neighbors (KNN)
- **Optimized Parameters**: n_neighbors = 4
- **Training Accuracy**: 98.95%
- **Test Accuracy**: 98.27%
- **F1-Score**: 98.67%

### 2. Logistic Regression
- **Training Accuracy**: 93.51%
- **Test Accuracy**: 93.24%
- **F1-Score**: 93.39%

### 3. Decision Tree Classifier
- **Optimized Parameters**: max_depth = 20, max_features = 5
- **Training Accuracy**: 100%
- **Test Accuracy**: 99.43%
- **F1-Score**: 99.41%

## 📈 Results

| Model | Precision | Recall | F1-Score | Training Time | Testing Time |
|-------|-----------|--------|----------|---------------|--------------|
| KNN | 98.64% | 98.31% | 98.67% | - | - |
| Logistic Regression | 93.57% | 94.31% | 93.39% | 0.098s | 0.005s |
| Decision Tree | 99.51% | 99.57% | 99.41% | 0.073s | 0.009s |

### Best Model: Decision Tree Classifier
The Decision Tree classifier achieved the highest performance with:
- 99.43% test accuracy
- Only 44 misclassifications out of 7,558 samples
- Excellent balance between precision (99.51%) and recall (99.57%)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
# Load and preprocess data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your dataset
train = pd.read_csv('Train_data.csv')
test = pd.read_csv('Test_data.csv')

# Run the complete pipeline
python main.py
```

### Training Individual Models

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=2)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=20, max_features=5)
dt.fit(X_train, y_train)

# Evaluate
score = dt.score(X_test, y_test)
print(f"Test Accuracy: {score}")
```

## 📁 Project Structure

```
network-intrusion-detection/
│
├── data/
│   ├── Train_data.csv
│   └── Test_data.csv
│
├── notebooks/
│   └── CS.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   └── evaluation.py
│
├── models/
│   └── saved_models/
│
├── results/
│   ├── figures/
│   └── reports/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 📦 Requirements

```
numpy>=1.21.6
pandas>=1.3.5
scikit-learn>=1.0.2
matplotlib>=3.5.3
seaborn>=0.12.2
lightgbm>=3.3.5
xgboost>=1.7.5
optuna>=3.0.3
tabulate>=0.9.0
```

## 🔬 Feature Selection

The project uses Recursive Feature Elimination (RFE) with Random Forest to select the top 10 most important features:

1. protocol_type
2. service
3. flag
4. src_bytes
5. dst_bytes
6. count
7. same_srv_rate
8. dst_host_srv_count
9. dst_host_same_srv_rate
10. dst_host_same_src_port_rate

## 🎓 Model Optimization

### Hyperparameter Tuning with Optuna

```python
import optuna

def objective(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10)
    
    classifier = DecisionTreeClassifier(
        max_features=dt_max_features,
        max_depth=dt_max_depth
    )
    classifier.fit(x_train, y_train)
    return classifier.score(x_test, y_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
```

## 📊 Visualization

The project includes comprehensive visualizations:
- Class distribution plots
- Model performance comparison charts
- Confusion matrices
- Precision-Recall curves
- F1-Score comparison bar charts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset source: [Add dataset source]
- Inspired by network security research and machine learning applications in cybersecurity
- Built with scikit-learn, Optuna, and other open-source libraries

## 📧 Contact

For questions or feedback, please reach out to [your-email@example.com]

---

**Note**: This is an educational project for demonstrating machine learning applications in network security. For production use, additional validation and security considerations are necessary.
