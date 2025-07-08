# 📉 Customer Churn Prediction

This project builds a machine learning model to predict whether a customer is likely to churn using the Telco Customer Churn dataset. It covers data cleaning, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and model explainability using LIME.

---

## 📁 Project Structure

```
.
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Customer_Churn_ML.ipynb
├── churn_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source**: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- The dataset includes information such as demographics, services signed up for, account information, and whether the customer churned.

---

## 🔍 Key Steps

1. **Data Cleaning**  
   - Handled missing values in `TotalCharges`  
   - Converted binary columns to boolean for clean EDA  
   - Encoded categorical features using one-hot encoding

2. **Exploratory Data Analysis (EDA)**  
   - Visualized churn distributions  
   - Analyzed churn by contract type and other features  
   - Correlation heatmap for numerical insights

3. **Modeling**  
   - Used **RandomForestClassifier**  
   - Applied **GridSearchCV** for hyperparameter tuning  
   - Evaluated using ROC-AUC, confusion matrix, and classification report

4. **Model Explainability**  
   - Used **LIME** to explain individual predictions with human-interpretable explanations

---

## 📈 Model Performance

- **Best ROC-AUC Score**: ~0.84  
- **Model**: Random Forest with tuned hyperparameters

---

## ✅ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `lime`
- `joblib`

---

## 🧠 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/customer-churn-ml.git
   cd customer-churn-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook Customer_Churn_ML.ipynb
   ```

4. Run the notebook cells to preprocess data, train the model, evaluate, and explain predictions.

---

## 🔍 Model Explainability

This project uses **LIME (Local Interpretable Model-Agnostic Explanations)** to understand individual predictions. LIME provides a visual explanation showing which features influenced the decision.

---

## 💾 Saving the Model

The trained model and scaler are saved as:

- `churn_model.pkl`
- `scaler.pkl`

These can be reused for inference without retraining.

---

## 📬 Contact

For questions or suggestions, feel free to reach out or create an issue.

---

## 📜 License

This project is open-source and available under the MIT License.
