# Financial Fraud Detection Machine Learning Pipeline

## 📌 Project Overview
This project aims to detect fraudulent financial transactions using advanced Machine Learning techniques. Financial fraud datasets are notoriously highly imbalanced, meaning legitimate transactions vastly outnumber fraudulent ones. To tackle this, the project implements a robust end-to-step pipeline, culminating in a highly optimized **XGBoost** classifier designed to maximize the detection of rare fraud events while keeping false alarms (false positives) under control.

## 📂 Dataset & Access
The data used in this project is a financial transaction dataset (`Fraud.csv`).

### How to Access the Data:
1. The dataset is hosted on Google Drive. You can access and download it via this link: **[Fraud.csv Dataset](https://drive.google.com/file/d/1kgeDuGx2VVyr0R4FB4-9JrLrUjPRA1XS/view?usp=sharing)**.
2. Download the `.csv` file and place it in the root directory of this project alongside the notebook before running the code.

### Data Insights:
* **Total transactions**: 6,362,620
* **Fraudulent transactions**: 8,213
* **Overall fraud rate**: ~0.1291%
* **Context**: The existing rule-based system flagged less than 0.001% of actual frauds, demonstrating the critical need for a predictive ML model. Fraud rates were also observed to be significantly higher during nighttime hours and for higher transaction amounts.

## ⚙️ Project Pipeline
The project follows a structured data science pipeline:

### 1. Exploratory Data Analysis (EDA)
* **Temporal Analysis**: Analyzed transaction patterns across different hours of the day, revealing a stark contrast in fraud rates between day and night.
* **Distribution Checks**: Visualized the highly skewed distribution of transaction amounts using log-scale histograms.

### 2. Feature Engineering & Data Cleaning
* **Multicollinearity Reduction**: Pre-transaction and post-transaction balances were highly correlated. Redundant columns (`oldbalanceOrg`, `newbalanceOrig`, etc.) were dropped and replaced with calculated features: `balance_diff_orig` and `balance_diff_dest`.
* **Outlier Flagging**: Created a new binary feature (`is_high_amount`) to flag transactions in the 99th percentile of amounts, capturing extreme values prone to fraud.
* **Categorical Encoding**: Transformed categorical transaction types into numerical formats using `LabelEncoder`.
* **Imputation**: Handled missing values to ensure model stability.

### 3. Data Splitting & Scaling
* **Stratified Splitting**: Split the data into 80% training and 20% testing sets, preserving the 0.129% fraud ratio in both sets.
* **Robust Scaling**: Applied `RobustScaler` to numerical columns to minimize the negative impact of extreme outliers.

### 4. Model Training & Evaluation
Three models were trained and compared:
1.  **Logistic Regression** (Baseline)
2.  **Random Forest**
3.  **XGBoost** (Gradient Boosting)

*Evaluation Metric*: Due to the massive class imbalance, **PR-AUC (Area Under the Precision-Recall Curve)** was used as the primary evaluation metric instead of standard accuracy or ROC-AUC.
* **Results**: XGBoost outperformed the others with an AUPRC of **0.875** (compared to Random Forest's 0.841 and Logistic Regression's 0.550).

### 5. Threshold Optimization & Fine-Tuning
* **Probability Threshold**: The default 0.5 classification threshold was heavily optimized using the Precision-Recall curve. By shifting the threshold to **0.9043**, the model achieves a **90% recall** (catching 90% of all frauds) while boosting precision to ~32.3%, significantly reducing costly false positives.
* **Hyperparameter Tuning**: The final XGBoost model was tuned to prevent overfitting and handle the imbalance:
  * `max_depth` = 4
  * `learning_rate` = 0.05
  * `min_child_weight` = 50
  * `scale_pos_weight` = Dynamically calculated ratio of negative to positive classes.

## 📁 Repository Structure
* `analysis.ipynb` / `analysis_2.ipynb`: Jupyter notebooks containing the full EDA, feature engineering, training, and evaluation pipeline.
* `best_fraud_model_tuned.pkl` / `best_fraud_model_tuned_2.pkl`: Serialized versions of the final, optimized XGBoost models ready for deployment or inference.
* `README.md`: Project documentation.

## 🛠️ Requirements & Dependencies
To run the notebook and use the model, you need the following Python libraries:
* `python 3.x`
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`
* `seaborn`
* `joblib`
