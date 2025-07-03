# ⚡ Household Power & Weather Dashboard

This Streamlit application visualizes and predicts household power consumption using both traditional machine learning models and a neural network. It integrates power usage data with weather information for insightful forecasting and analysis.

---

## 📦 Features

- 📈 **Exploratory Data Analysis**: Visualize daily and hourly trends, sub-metering statistics, and total power distribution.
- 🔍 **Outlier Detection**: Z-score-based filtering to improve data quality.
- 🌡️ **Weather Integration**: Correlates temperature with energy usage.
- 📊 **Correlation Heatmap**: Understand relationships between features.
- 🤖 **Machine Learning Models**:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- 🧠 **Neural Network (PyTorch)**:
  - 4-layer fully connected model
  - Adam optimizer, MSE loss

---

## 📁 File Structure

├── power_dashboard.py # Main Streamlit dashboard script
├── household_power_consumption.txt # Raw power dataset
├── temprature data.csv # Weather dataset
├── power_net.pth # (Optional) Saved PyTorch model
└── README.md # Project overview

📌 Model Overview
Machine Learning
Each ML model is trained on power + weather data:

train_test_split is used (75% train, 25% test)

Models evaluated using R² score and Mean Squared Error (MSE)

Neural Network (PyTorch)
Input: Scaled features

Architecture: [Input → 64 → 32 → 16 → 1]

Activation: ReLU + Leaky ReLU

Optimizer: Adam (lr=0.0001)

Loss: MSELoss
