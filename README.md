# 🌱 Soil Nutrient & Crop Predictor

This project implements a machine learning pipeline to determine which soil feature—**Nitrogen (N)**, **Phosphorous (P)**, **Potassium (K)**, or **pH**—is the most reliable indicator for predicting crop types. 

## 📖 Overview
In precision agriculture, understanding which soil metric holds the most "predictive power" is essential for optimizing land use. This script performs a univariate analysis by training individual **Logistic Regression** models for each soil nutrient and ranking them based on their **Weighted F1-score**.

## ✨ Features
* **Feature Engineering:** Extracts individual soil metrics to test predictive strength in isolation.
* **Multinomial Logistic Regression:** Uses `scikit-learn` to handle multi-class crop classification.
* **Performance Ranking:** Automates the calculation of F1-scores for every feature and identifies the "Best Predictive Feature."
* **Data Visualization Ready:** Outputs clear performance metrics for N, P, K, and pH.

## 🛠️ Tech Stack
* **Python**
* **Pandas & NumPy** (Data processing)
* **Scikit-learn** (Model training, splitting, and metrics)

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ajlrza/soil-nutrient-predictor.git](https://github.com/ajlrza/soil-nutrient-predictor.git)
   cd soil-nutrient-predictor
   ```

2. **Ensure your dataset is ready:**
   Place your `soil_measures.csv` file in the root directory.

3. **Run the script:**
   ```bash
   python soil_analysis.py
   ```

## 📊 Evaluation Logic
The script iterates through the features and calculates the **Weighted F1-score**, which balances precision and recall across all crop classes. 

```python
# Example Output:
F1-score for N: 0.38
F1-score for P: 0.25
F1-score for K: 0.22
F1-score for ph: 0.09
# Best Predictive Feature: {'N': 0.38}
```

---
*Developed as a project in Machine Learning for Environmental Science.*
