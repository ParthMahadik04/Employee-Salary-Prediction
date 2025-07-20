# 🧠 Employee Salary Classification using Machine Learning

> ✅ This project was developed during the **EDUNET FOUNDATION - IBM SkillsBuild Artificial Intelligence 6-week Internship**.  
> It involved end-to-end development of an AI system that predicts whether an employee salary, based on demographic and work-related attributes.

---

## 📌 Project Objective

To build a machine learning model that can **predict the salary of an individual** using structured data from the Dataset. The prediction is based on various features such as age, education, occupation, hours-per-week, and more.

---

## 🛠️ Technologies Used

- **Python 3**
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib**
- **Streamlit** (for interactive UI)
- **Joblib** (for model export)
- **Git & GitHub** (for version control)

---

## 📊 Dataset Used

- **Dataset was provided by the mentors**
- Contains over 48,000 records
- Features include: age, workclass, education, occupation, hours-per-week, etc.
- Target label: `<=50K` or `>50K` annual income

---

## ✅ Steps Followed

### 1. Import Required Libraries
We used core Python libraries like:
- `pandas` for data manipulation
- `scikit-learn` for machine learning models and metrics
- `matplotlib` for visualizations
- `joblib` for model serialization

### 2. Load the Dataset
The dataset was loaded using `pandas.read_csv()` from a file named `adult 3.csv`.

### 3. Data Cleaning
- Removed unnecessary columns like `fnlwgt` and `educational-num`.
- Handled missing values represented by `?`.
- Converted salary labels into binary format (`<=50K`, `>50K`).

### 4. Data Sampling
To reduce processing time, stratified sampling (30%) was used to maintain class balance between high-income and low-income groups.

### 5. Feature Engineering
- Converted categorical variables into numerical format using **one-hot encoding**.
- Extracted features and labels for training.

### 6. Train-Test Split
Used `train_test_split` with stratification to maintain class distribution across training and testing sets.

### 7. Model Building
Two models were trained and evaluated:
- 🎯 **Random Forest Classifier**
- 🚀 **Gradient Boosting Classifier**

### 8. Model Evaluation
Each model was evaluated on accuracy. The best performing model was selected based on the highest accuracy score.

### 9. Visualization
Used `matplotlib` to generate a **bar chart comparison** of model accuracies. The best-performing model was highlighted with a different color.

### 10. Model Saving
The best model and its feature list were saved using `joblib.dump()` into a file named `mymodel.pkl`.



