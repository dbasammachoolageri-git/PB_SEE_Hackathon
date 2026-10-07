# PB_SEE_Hackathon
# 🛡️ Network Intrusion Detection

### Normal vs Attack Traffic Using Machine Learning

A machine-learning based **Network Intrusion Detection** prototype that classifies network traffic into **Normal** or **Attack** using the **NSL-KDD dataset**.

The project compares three machine-learning models — **Decision Tree, Random Forest, and Logistic Regression** — and selects **Random Forest** as the final model based on its performance.

---

## 📌 Project Overview

Computer networks continuously generate large amounts of traffic. Some network connections may represent suspicious or unauthorized activity such as denial-of-service attacks, probing, scanning, and password-guessing attempts.

The objective of this project is to develop a machine-learning classifier that can automatically identify whether a network connection is:

- 🟢 **Normal Traffic**
- 🔴 **Attack Traffic**

The project also analyzes feature importance and evaluates detection performance across different attack categories.

---

## 🎯 Objectives

- Detect network traffic as **Normal or Attack**
- Preprocess and analyze the NSL-KDD dataset
- Compare multiple machine-learning algorithms
- Select the best-performing classification model
- Analyze the confusion matrix
- Identify important network features
- Evaluate detection rates for different attack categories
- Demonstrate predictions using Google Colab

---

## 📊 Dataset

The project uses the **NSL-KDD dataset**, a benchmark dataset commonly used for network intrusion detection research.

| Dataset Information | Value |
|---|---:|
| Total Records | 125,973 |
| Normal Records | 67,343 |
| Attack Records | 58,630 |
| Training Samples | 100,778 |
| Testing Samples | 25,195 |
| Missing Values | 0 |

The original attack labels were retained for attack-type analysis, while the classification task converts the labels into two classes:

**Normal → Normal Traffic**

**All Attack Types → Attack Traffic**

---

## 🔄 Project Workflow

```text
NSL-KDD Dataset
       ↓
Data Preprocessing
       ↓
Label Conversion
       ↓
Normal vs Attack Classification
       ↓
Model Training
       ↓
Model Comparison
       ↓
Random Forest Selection
       ↓
Prediction & Evaluation
       ↓
Feature Importance & Attack Analysis
```

---

## 🤖 Machine Learning Models

Three models were evaluated:

1. **Random Forest**
2. **Decision Tree**
3. **Logistic Regression**

### Model Comparison

| Model | Accuracy | Attack Precision | Attack Recall | Attack F1 |
|---|---:|---:|---:|---:|
| **Random Forest** | **99.90%** | **99.95%** | **99.85%** | **99.90%** |
| Decision Tree | 99.86% | 99.82% | 99.87% | 99.85% |
| Logistic Regression | 97.18% | 97.64% | 96.27% | 96.95% |

### 🏆 Selected Model

**Random Forest** was selected as the final model because it achieved the highest **Attack-class F1-score** among the evaluated models.

**Final Test Accuracy: 99.9047%**

> These results are measured on the NSL-KDD test split used in this project and should not be interpreted as universal real-world accuracy.

---

## 📈 Confusion Matrix

|  | Actual Normal | Actual Attack |
|---|---:|---:|
| Predicted Normal | 13,463 (TN) | 18 (FN) |
| Predicted Attack | 6 (FP) | 11,708 (TP) |

### Results

- **True Negative (TN):** 13,463 normal connections correctly classified
- **False Positive (FP):** 6 normal connections incorrectly classified as attacks
- **False Negative (FN):** 18 attack connections incorrectly classified as normal
- **True Positive (TP):** 11,708 attack connections correctly detected

---

## 🔍 Feature Importance

The Random Forest model identified the following features as the most important:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `src_bytes` | 14.59% |
| 2 | `flag_SF` | 8.53% |
| 3 | `dst_host_same_srv_rate` | 8.25% |
| 4 | `dst_bytes` | 7.51% |
| 5 | `dst_host_srv_count` | 5.71% |
| 6 | `diff_srv_rate` | 3.65% |
| 7 | `count` | 3.50% |
| 8 | `same_srv_rate` | 3.37% |
| 9 | `protocol_type_icmp` | 3.30% |
| 10 | `dst_host_diff_srv_rate` | 2.99% |

`src_bytes` had the highest individual feature importance in this experiment.

---

## 🚨 Attack-Type Analysis

The project also evaluated detection performance across the original attack categories.

| Category | Test Samples | Detected | Detection Rate |
|---|---:|---:|---:|
| R2L | 226 | 215 | 95.13% |
| Probe | 2,389 | 2,383 | 99.75% |
| DoS | 9,105 | 9,104 | 99.99% |
| U2R | 6 | 6 | 100.00% |

The weakest broad category in this test was **R2L**, with a detection rate of **95.13%**.

Among attack types with at least 10 test samples, `guess_passwd` had the lowest detection rate at **92.31%**.

> Rare attack types contain very few test samples, so their detection percentages can be unstable.

---

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Programming and machine-learning implementation |
| **Google Colab** | Development and execution environment |
| **Pandas** | Data loading and preprocessing |
| **Scikit-learn** | Machine-learning models and evaluation |
| **Matplotlib** | Data visualization |

---

## 🧪 Live Demonstration

The working prototype is implemented in **Google Colab**.

The trained Random Forest model receives network-connection features and predicts whether the traffic is Normal or Attack.

| Demo | Input | Observed Output |
|---|---|---|
| 1 | Real normal test connection | Normal Traffic — 0.00% attack probability |
| 2 | Real attack test connection | Attack Traffic — 100.00% attack probability |
| 3 | Illustrative flood-like connection | Attack Traffic — 95.00% attack probability |
| 4 | Modified normal connection | Normal Traffic — 9.00% attack probability |

---

## 📁 Project Structure

```text
Network-Intrusion-Detection/
│
├── README.md
├── Network_Intrusion_Detection.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
└── report/
    └── Network_Intrusion_Detection_Report.pdf
```

*The exact files and folders may vary depending on the files uploaded to the repository.*

---

## ⚠️ Limitations

- The prototype is evaluated using the **NSL-KDD dataset**, not live network traffic.
- Reported accuracy represents performance on the project's test data.
- Real-world deployment would require testing on current network traffic.
- Additional and more recent intrusion-detection datasets should be considered.
- Rare attack categories have limited samples.
- Real deployment would require monitoring, retraining, and additional security considerations.

---

## 🚀 Future Scope

- Integrate the classifier with real-time network traffic monitoring
- Improve detection of rare attack categories
- Evaluate larger and more recent intrusion-detection datasets
- Develop a web-based dashboard for visualization and monitoring
- Explore advanced machine-learning and deep-learning techniques

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to classify network traffic into **Normal** and **Attack** categories.

Among the three evaluated models, **Random Forest** achieved the strongest performance, with:

**99.90% Accuracy**

**99.90% Attack-class F1-score**

on the project's NSL-KDD test set.

Feature-importance and attack-type analysis provide additional insight into the model's behavior and highlight areas for future improvement.

---

## 👨‍💻 Project Information

**Project:** Network Intrusion Detection  
**Classification:** Normal vs Attack  
**Dataset:** NSL-KDD  
**Best Model:** Random Forest  
**Development Environment:** Google Colab  
**Institution:** REVA University  
**Event:** Hackathon – Abhinava

---

⭐ **If you find this project useful, consider giving the repository a star!**
