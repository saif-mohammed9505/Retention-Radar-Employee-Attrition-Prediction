# 🚀 Retention Radar

### 🧑‍💼 Employee Attrition Prediction & Retention Risk Analysis

**Retention Radar** is an interactive machine learning application for predicting **employee attrition** and analyzing retention risk.

The project uses a **Bagging Ensemble Classifier with Decision Trees** and provides a Streamlit dashboard for both **individual employee assessment** and **batch employee analysis through CSV upload**.

---

## 🌟 About the Project

Employee attrition can create challenges for organizations such as increased recruitment costs, loss of experienced employees, and team instability.

Retention Radar uses employee HR information to estimate whether an employee is likely to leave the organization.

The application supports two prediction modes:

- 👤 **Single Employee** — assess one employee using an interactive profile.
- 📂 **Batch Upload** — upload multiple employee records through a CSV file and score them together.

The goal is to provide a simple interface for exploring machine learning-based employee attrition predictions.

---

## ✨ Key Features

### 👤 Single Employee Assessment

Users can enter an employee's information through an interactive dashboard.

The assessment considers:

- 😊 Satisfaction level
- 📊 Last evaluation score
- 📁 Number of projects
- ⏱️ Average monthly hours
- 🏢 Years spent at the company
- ⚠️ Work accident history
- 📈 Promotion in the last 5 years
- 🏷️ Department
- 💰 Salary band

After assessment, the application provides:

- 🎯 Attrition probability
- 🚦 Risk status
- 🔎 Notable risk signals
- 📋 Employee profile summary

---

### 📂 Batch Employee Upload

Retention Radar can also analyze **multiple employees at once**.

Users can upload a CSV file containing employee records and preview the uploaded data before scoring it.

The batch workflow allows users to:

- 📤 Upload employee data
- 👀 Preview the uploaded records
- ▶️ Score all rows
- 🎯 Generate attrition probability for every employee
- 🔮 Generate predicted attrition results
- 🚨 Identify high-risk employees
- 📊 View the average risk across all uploaded employees
- 📋 Review the complete scored dataset

This makes the application suitable for analyzing a group of employees without manually entering each employee profile.

---

### 📊 Batch Risk Summary

After scoring the uploaded dataset, the application displays a quick summary of the results.

The dashboard includes:

| Metric | Description |
|---|---|
| 👥 **Rows scored** | Total number of employee records processed |
| 🚨 **Flagged high risk (≥50%)** | Number of employees whose attrition probability is at least 50% |
| 📊 **Average risk** | Average attrition probability across the uploaded employees |

The results table also provides:

- Employee input features
- 🎯 Attrition probability
- 🔮 Predicted left/stay result

---

### 🧠 Model Insights

The application provides information about the trained machine learning model and its ensemble configuration.

The project uses **20 decision-tree estimators** inside the Bagging ensemble.

---

### ℹ️ About Section

The application includes an About section describing the project, its purpose, and the employee attrition prediction workflow.

---

## 🤖 Machine Learning Model

The project uses a **Bagging Classifier** with **Decision Trees** as the base estimators.

### 🌳 How Bagging Works

Bagging, or Bootstrap Aggregating, trains multiple models using different bootstrap samples of the training data.

The predictions from the individual decision trees are then combined to produce the final classification.

This approach helps create a more stable ensemble than relying on a single decision tree.

### ⚙️ Model Configuration

| Parameter | Configuration |
|---|---|
| 🤖 Model | Bagging Classifier |
| 🌳 Base estimator | Decision Tree Classifier |
| 🌲 Number of estimators | 20 |
| 📐 Criterion | Gini |
| 🌿 Maximum tree depth | 4 |
| 🔄 OOB score | Enabled |
| ⚡ Parallel processing | Enabled |
| 🎲 Random state | 400 |

---

## 📋 Input Features

The model uses the following employee attributes:

| Feature | Description |
|---|---|
| 😊 `satisfaction_level` | Employee satisfaction level |
| 📊 `last_evaluation` | Most recent evaluation score |
| 📁 `number_project` | Number of projects handled |
| ⏱️ `average_montly_hours` | Average monthly working hours |
| 🏢 `time_spend_company` | Number of years spent at the company |
| ⚠️ `Work_accident` | Whether the employee experienced a work accident |
| 📈 `promotion_last_5years` | Whether the employee received a promotion in the last 5 years |
| 🏷️ `dept` | Employee department |
| 💰 `salary` | Employee salary category |

The target variable is:

- `0` → Employee predicted to stay
- `1` → Employee predicted to leave

---

## 🔄 Data Preprocessing

The machine learning workflow includes several preprocessing steps:

### 🧹 Data Cleaning

- Loaded the HR dataset using Pandas.
- Checked for duplicate records.
- Removed duplicate records.
- Checked for missing values.
- Examined numerical variables and their distributions.
- Investigated potential outliers.

### 📌 Outlier Handling

The project examines outliers using the **IQR method**.

Values of `time_spend_company` above `5.5` are replaced with the mean of that feature.

### 🔢 Encoding

The salary categories are converted into numerical values:

| Salary | Encoded Value |
|---|---:|
| Low | 0 |
| Medium | 1 |
| High | 2 |

The department feature is transformed using **one-hot encoding**.

### ✂️ Train/Test Split

The processed dataset is divided into training and testing sets using an **80/20 split**.

---

## 🎯 Prediction Output

The application performs binary classification.

| Prediction | Meaning |
|---|---|
| 🟢 `0` | Employee predicted to stay |
| 🔴 `1` | Employee predicted to leave |

In addition to the class prediction, the application displays an **attrition probability**, making it possible to identify employees with higher estimated risk.

---

## 📈 Model Performance

The trained Bagging model achieved:

### 🏆 Test Accuracy

**97.17%**

The evaluation was performed on **2,399 test samples**.

### 📊 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 🟢 Stay (0) | 0.98 | 0.99 | 0.98 | 2009 |
| 🔴 Leave (1) | 0.95 | 0.87 | 0.91 | 390 |
| **Accuracy** | | | **0.97** | **2399** |
| Macro Average | 0.96 | 0.93 | 0.95 | 2399 |
| Weighted Average | 0.97 | 0.97 | 0.97 | 2399 |

---

## 🔄 Out-of-Bag Evaluation

The Bagging model uses **Out-of-Bag (OOB) evaluation**.

In bagging, each decision tree is trained using a bootstrap sample. Some training records are not selected for a particular tree.

Those unused records can be used to estimate the model's out-of-bag performance.

The project also explores different numbers of estimators and uses OOB error as a way to evaluate ensemble performance.

---

## 🔍 Feature Importance

The project examines feature importance from the decision trees inside the Bagging ensemble.

This helps understand which input variables contribute to the decisions made by the individual tree estimators.

---

## 💾 Trained Model

The trained model is saved as:

**`employee_attrition_bagging.pkl`**

The saved model can be loaded by the Streamlit application to perform predictions without retraining the model every time the application starts.

---

## 🖥️ Application Workflow

### 👤 Single Employee

**Employee Profile → Input Features → Bagging Model → Attrition Probability → Risk Read-out**

### 📂 Batch Analysis

**CSV Upload → Data Preview → Score All Rows → Attrition Probability → Risk Summary → Scored Results**

---

## 📊 Example Batch Result

A batch analysis can provide a summary such as:

| 📌 Metric | Example Result |
|---|---:|
| 👥 Rows scored | 10 |
| 🚨 Flagged high risk (≥50%) | 5 |
| 📊 Average risk | 50.5% |

Each scored employee can then be reviewed using the generated probability and predicted outcome.

---

## 🛠️ Technologies Used

- 🐍 **Python**
- 🐼 **Pandas**
- 🔢 **NumPy**
- 🤖 **Scikit-learn**
- 💾 **Joblib**
- 🎨 **Streamlit**
- 📓 **Jupyter Notebook**
- 📊 **Matplotlib**
- 📈 **Seaborn**

---

## 📁 Project Structure

```text
retention_radar/
│
├── .streamlit/
├── assets/
├── screenshots/
│
├── app.py
├── ensemble bagging.ipynb
├── employee_attrition_bagging.pkl
├── sample_employees.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Running the Project

### 1️⃣ Install the dependencies

Install all required Python packages from the project's `requirements.txt` file.

### 2️⃣ Start the Streamlit application

Launch `app.py` using Streamlit.

### 3️⃣ Use the application

You can then:

- 👤 Assess a single employee
- 📂 Upload a CSV containing multiple employees
- ▶️ Score the uploaded records
- 📊 Review the batch risk summary
- 🔎 Inspect individual prediction results
- 🧠 View model information

---

## 🧪 Sample Data

The project includes:

**`sample_employees.csv`**

This file can be used to test the batch-upload functionality and understand the expected employee data format.

---

## 🎯 Project Highlights

- 🤖 Machine learning-based employee attrition prediction
- 🌳 Bagging ensemble using Decision Trees
- 📊 97.17% test accuracy
- 👤 Individual employee prediction
- 📂 Multi-employee CSV batch prediction
- 🚨 High-risk employee identification
- 📈 Average batch risk analysis
- 🔍 Feature importance analysis
- 🔄 Out-of-Bag evaluation
- 🎨 Interactive Streamlit dashboard
- 💾 Saved trained model for application use

---

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes**.

Employee attrition predictions should not be used as the sole basis for employment decisions. Predictions should be interpreted carefully and considered together with appropriate human review and organizational context.

---

## 👨‍💻 Project Information

| | |
|---|---|
| 📌 **Project Name** | Retention Radar |
| 🎯 **Task** | Employee Attrition Prediction |
| 🤖 **Model** | Bagging Classifier |
| 🌳 **Base Model** | Decision Tree |
| 📊 **Problem Type** | Binary Classification |
| 🖥️ **Interface** | Streamlit |
| 🐍 **Language** | Python |

---

### ⭐ Retention Radar

**Predict attrition. Understand risk. Support better retention analysis.**

## 👨‍💻 Author

**Saif Mohammed**

GitHub:

https://github.com/saif-mohammed9505

---

## License

This project is intended for educational and experimental purposes.
