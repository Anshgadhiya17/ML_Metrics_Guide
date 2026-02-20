# 📘 Supervised Learning – Complete Guide (With Models, Terms & Examples)

Supervised Learning is a type of Machine Learning where:

👉 We have input data (X)  
👉 We have output/target labels (y)  
👉 Model learns mapping from X → y  

It learns from labeled data.

---

# 📌 Example of Supervised Learning

✔ Predict house price  
✔ Spam detection  
✔ Disease prediction  
✔ Student pass/fail  
✔ Sales forecasting  

---

# 🧠 Types of Supervised Learning

1️⃣ Regression  
2️⃣ Classification  

---

# 🔹 1️⃣ Regression

Used when output is continuous numeric value.

Examples:
- House price prediction
- Salary prediction
- Temperature prediction

---

# 🔹 2️⃣ Classification

Used when output is category/class.

Examples:
- Spam or Not Spam
- Yes or No
- Cat, Dog, Bird

---

# 📊 Important Terms in Supervised Learning

## 🔹 Feature (Independent Variable)
Input variables.

Example:
Area, bedrooms → house price model

---

## 🔹 Target (Dependent Variable)
Output variable.

Example:
House price

---

## 🔹 Training Data
Data used to train model.

---

## 🔹 Testing Data
Data used to evaluate model.

---

## 🔹 Overfitting
Model performs very well on training data  
But poor on testing data.

---

## 🔹 Underfitting
Model performs poorly on both training & testing data.

---

## 🔹 Bias
Error due to wrong assumptions.

---

## 🔹 Variance
Error due to model complexity.

---

## 🔹 Loss Function
Measures how wrong predictions are.

---

## 🔹 Accuracy
Percentage of correct predictions.

---

# 📈 Regression Models

---

# 🔹 Linear Regression

Simple linear relationship between X and y.

Equation:
y = mx + b

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

Use when:
✔ Data is linear  
✔ Simple problem  

---

# 🔹 Polynomial Regression

Used when relationship is curved.

---

# 🔹 Decision Tree Regressor

Splits data based on conditions.

Good for:
✔ Non-linear data  
✔ Easy interpretation  

---

# 🔹 Random Forest Regressor

Collection of multiple decision trees.

Advantages:
✔ High accuracy  
✔ Reduces overfitting  

---

# 📊 Classification Models

---

# 🔹 Logistic Regression

Used for binary classification.

Output between 0 and 1 (probability).

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

# 🔹 K-Nearest Neighbors (KNN)

Classifies based on nearest neighbors.

Parameter:
K = Number of neighbors

---

# 🔹 Decision Tree Classifier

Tree-based classification.

---

# 🔹 Random Forest Classifier

Multiple decision trees combined.

Very powerful & commonly used.

---

# 🔹 Support Vector Machine (SVM)

Finds best boundary (hyperplane).

Works well in:
✔ High dimensional data  
✔ Text classification  

---

# 🔹 Naive Bayes

Based on probability theorem.

Used in:
✔ Spam detection  
✔ Text classification  

---

# 📊 Evaluation Metrics (Regression)

| Metric | Meaning |
|---------|----------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Goodness of fit |

Example:

```python
from sklearn.metrics import mean_squared_error
```

---

# 📊 Evaluation Metrics (Classification)

| Metric | Meaning |
|---------|----------|
| Accuracy | Correct predictions |
| Precision | True Positive / Predicted Positive |
| Recall | True Positive / Actual Positive |
| F1-Score | Harmonic mean of Precision & Recall |
| Confusion Matrix | Detailed result table |

---

# 📉 Confusion Matrix

| Actual \ Predicted | Positive | Negative |
|---------------------|----------|----------|
| Positive | TP | FN |
| Negative | FP | TN |

---

# 📊 Bias-Variance Tradeoff

High Bias → Underfitting  
High Variance → Overfitting  

Goal:
Find balance between both.

---

# 📈 Model Training Process

1. Collect Data  
2. Clean Data  
3. Split Data (Train/Test)  
4. Train Model  
5. Evaluate Model  
6. Tune Hyperparameters  
7. Deploy Model  

---

# 📌 Cross Validation

Used to evaluate model better.

Example:

```python
from sklearn.model_selection import cross_val_score
```

---

# 📌 Hyperparameters

Parameters set before training.

Examples:
- K in KNN
- Depth in Decision Tree
- Learning rate

---

# 📊 Supervised vs Other Learning

| Feature | Supervised | Unsupervised | Reinforcement |
|----------|------------|--------------|---------------|
| Labels | Yes | No | No |
| Output | Predict value | Find patterns | Reward-based |
| Example | Spam detection | Clustering | Game AI |

---

# 🎯 When to Use Which Model?

| Problem | Recommended Model |
|----------|-------------------|
| Linear data | Linear Regression |
| Non-linear data | Random Forest |
| Binary classification | Logistic Regression |
| High accuracy needed | Random Forest |
| Text data | Naive Bayes / SVM |
| Small dataset | KNN |

---

# 🚀 Final Summary

✔ Supervised learning uses labeled data  
✔ Two types: Regression & Classification  
✔ Many algorithms available  
✔ Need evaluation metrics  
✔ Avoid overfitting  
✔ Tune hyperparameters  

Supervised Learning = Learn from labeled examples
