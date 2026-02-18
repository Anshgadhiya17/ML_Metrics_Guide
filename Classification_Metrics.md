# 📊 Classification Metrics (scikit-learn)

Classification metrics are used to evaluate models that predict **categorical labels**.

Example:
- Spam / Not Spam
- Disease / No Disease
- Pass / Fail

---

# 📌 1️⃣ Confusion Matrix

A Confusion Matrix shows how many predictions were correct and incorrect.

|                | Predicted Positive | Predicted Negative |
|---------------|-------------------|-------------------|
| Actual Positive | TP (True Positive) | FN (False Negative) |
| Actual Negative | FP (False Positive) | TN (True Negative) |

### Definitions:

- **TP** → Correctly predicted Positive
- **TN** → Correctly predicted Negative
- **FP** → Incorrectly predicted Positive
- **FN** → Incorrectly predicted Negative

---

## 🧠 Small Real Example

Suppose we built a model to detect Disease.

Out of 10 patients:

- 4 actually have disease
- 6 do not have disease

Model predictions result:

- TP = 3  
- FN = 1  
- FP = 2  
- TN = 4  

Confusion Matrix:

|                | Predicted Yes | Predicted No |
|---------------|--------------|-------------|
| Actual Yes    | 3            | 1           |
| Actual No     | 2            | 4           |

---

# 📌 2️⃣ Accuracy

### 📖 Definition:
Accuracy tells how many predictions were correct overall.

### 🧮 Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

### 📌 Using Example:

Accuracy = (3 + 4) / 10  
Accuracy = 7 / 10 = **0.70 (70%)**

---

# 📌 3️⃣ Precision

### 📖 Definition:
Out of all predicted positives, how many were actually positive?

### 🧮 Formula:

Precision = TP / (TP + FP)

### 📌 Using Example:

Precision = 3 / (3 + 2)  
Precision = 3 / 5 = **0.60 (60%)**

👉 Important when **False Positives are costly**
Example: Spam detection

---

# 📌 4️⃣ Recall (Sensitivity)

### 📖 Definition:
Out of all actual positives, how many were correctly predicted?

### 🧮 Formula:

Recall = TP / (TP + FN)

### 📌 Using Example:

Recall = 3 / (3 + 1)  
Recall = 3 / 4 = **0.75 (75%)**

👉 Important when **False Negatives are costly**
Example: Disease detection

---

# 📌 5️⃣ F1 Score

### 📖 Definition:
Harmonic mean of Precision and Recall.

Used when we need balance between Precision and Recall.

### 🧮 Formula:

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

### 📌 Using Example:

F1 = 2 × (0.60 × 0.75) / (0.60 + 0.75)  
F1 = 0.67

---

# 📌 6️⃣ Support

Support = Number of actual occurrences of each class in dataset.

---

# 📌 7️⃣ scikit-learn Implementation

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = [1,1,1,1,0,0,0,0,0,0]
y_pred = [1,1,1,0,1,0,0,0,1,0]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
