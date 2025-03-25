# Customer Churn Prediction using XGBoost

## 📌 Overview
This project is an AI-powered web application built with **Streamlit**, utilizing **XGBoost** to predict customer churn. It analyzes customer data to determine whether a customer will stay or leave a company based on financial and demographic details.

## 🚀 Features
- Interactive UI with **Streamlit**
- Real-time customer churn prediction
- Uses **MinMaxScaler** for feature scaling
- Displays probability of churn and retention
- Provides customer insights & key influencing factors

## 🛠️ Tech Stack
- **Frontend**: Streamlit (Python-based UI)
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Model Deployment**: Pickle (Serialized Model)

## 📂 Project Structure
```
├── app.py  # Streamlit app for prediction
├── churn-prediction-using-xgboost.ipynb  # Jupyter Notebook (model training)
├── best_model.pkl  # Trained XGBoost model
├── scaler.pkl  # MinMaxScaler for data normalization
├── Churn_Modelling.csv  # Dataset used for training
└── README.md  # Project Documentation
```

## 📦 Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/xenon1919/Churn-Prediction.git
   cd Churn-Prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage
1. Adjust customer details using the sidebar options.
2. Click **Predict 🚀** to analyze the data.
3. View **Predicted Status** and probability of churn/retention.

## 📊 Dataset Information
The dataset includes:
- **Demographic Features** (Age, Geography, Gender)
- **Financial Features** (Credit Score, Balance, Salary)
- **Account Details** (Tenure, Number of Products, Activity Status)

## 📌 Model & Methodology
- **Model**: XGBoost Classifier
- **Preprocessing**:
  - Missing values handled
  - Categorical encoding (One-Hot Encoding)
  - Feature scaling using MinMaxScaler
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score

## 📜 License
This project is open-source under the **MIT License**.

## 🤝 Contributing
Feel free to submit pull requests for improvements!

## 🌟 Acknowledgements
Special thanks to the **Streamlit** and **XGBoost** communities!

---

Made with ❤️ by [Ramanchi Rishi Sai Teja]

