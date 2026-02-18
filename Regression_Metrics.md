# 📈 Regression Metrics (scikit-learn)

Regression metrics are used to evaluate models that predict **continuous values**.

Example:
- House price prediction
- Salary prediction
- Temperature prediction

---

# 📌 1️⃣ Mean Absolute Error (MAE)

### 📖 Definition:
Average of absolute differences between actual and predicted values.

It tells: "On average, how much error is there?"

### 🧮 Formula:

MAE = (1/n) × Σ |yᵢ − ŷᵢ|

Where:
- yᵢ = actual value
- ŷᵢ = predicted value
- n = total samples

---

## 🧠 Small Real Example

Suppose actual house prices (in lakhs):

y_true = [100, 200, 300]

Predicted:

y_pred = [110, 190, 310]

Errors:

| Actual | Predicted | Absolute Error |
|--------|-----------|---------------|
| 100    | 110       | 10            |
| 200    | 190       | 10            |
| 300    | 310       | 10            |

MAE = (10 + 10 + 10) / 3  
MAE = 10

👉 On average model is wrong by 10 lakhs.

---

# 📌 2️⃣ Mean Squared Error (MSE)

### 📖 Definition:
Average of squared differences between actual and predicted values.

Large errors get more penalty.

### 🧮 Formula:

MSE = (1/n) × Σ (yᵢ − ŷᵢ)²

Using same example:

Squared Errors:

10² = 100  
10² = 100  
10² = 100  

MSE = (100 + 100 + 100) / 3  
MSE = 100

👉 Punishes large mistakes more.

---

# 📌 3️⃣ Root Mean Squared Error (RMSE)

### 📖 Definition:
Square root of MSE.

Gives error in same unit as target variable.

### 🧮 Formula:

RMSE = √MSE

Using example:

RMSE = √100  
RMSE = 10

👉 Easier to interpret than MSE.

---

# 📌 4️⃣ R² Score (Coefficient of Determination)

### 📖 Definition:
Measures how well model explains variance of data.

Range:
- 1 → Perfect model
- 0 → No improvement over mean
- Negative → Very bad model

### 🧮 Formula:

R² = 1 − (SS_res / SS_total)

Where:
- SS_res = Σ (yᵢ − ŷᵢ)²
- SS_total = Σ (yᵢ − ȳ)²
- ȳ = mean of actual values

---

## 🧠 Simple Understanding

If:
- R² = 0.90 → Model explains 90% variance
- R² = 0.50 → Model explains 50% variance

Higher is better.

---

# 📌 5️⃣ scikit-learn Implementation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [100, 200, 300]
y_pred = [110, 190, 310]

print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R2 Score:", r2_score(y_true, y_pred))
