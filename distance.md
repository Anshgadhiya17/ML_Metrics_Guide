# 📏 Distance Metrics in Machine Learning (With Formula & Example)

Distance metrics are used to measure similarity between two data points.

Smaller distance → More similar  
Larger distance → Less similar  

Used in:
✔ KNN  
✔ K-Means  
✔ Clustering  
✔ Recommendation systems  

---

# 📌 Suppose We Have Two Points:

A = (2, 3)  
B = (5, 7)

We will calculate different distances between these two points.

---

# 🔹 1️⃣ Euclidean Distance (Most Common)

Straight-line distance between two points.

## 📐 Formula:

For 2D:

d = √((x2 - x1)² + (y2 - y1)²)

General Formula:

d = √Σ (xi - yi)²

---

## 🧮 Example:

A = (2, 3)  
B = (5, 7)

d = √((5-2)² + (7-3)²)  
d = √(3² + 4²)  
d = √(9 + 16)  
d = √25  
d = 5  

✔ Euclidean Distance = 5

---

# 🔹 2️⃣ Manhattan Distance

Distance measured along grid lines (like city blocks).

Also called:
L1 Distance

## 📐 Formula:

d = |x2 - x1| + |y2 - y1|

General:

d = Σ |xi - yi|

---

## 🧮 Example:

A = (2, 3)  
B = (5, 7)

d = |5-2| + |7-3|  
d = 3 + 4  
d = 7  

✔ Manhattan Distance = 7

---

# 🔹 3️⃣ Minkowski Distance

General form of Euclidean & Manhattan.

## 📐 Formula:

d = ( Σ |xi - yi|^p )^(1/p)

Where:
p = 1 → Manhattan  
p = 2 → Euclidean  

---

## 🧮 Example (p = 3):

d = (|3|³ + |4|³)^(1/3)  
d = (27 + 64)^(1/3)  
d = 91^(1/3)

---

# 🔹 4️⃣ Chebyshev Distance

Maximum absolute difference in any dimension.

## 📐 Formula:

d = max(|xi - yi|)

---

## 🧮 Example:

A = (2,3)  
B = (5,7)

|5-2| = 3  
|7-3| = 4  

Max value = 4  

✔ Chebyshev Distance = 4

---

# 🔹 5️⃣ Cosine Distance

Measures angle between two vectors.

Used in:
✔ Text similarity  
✔ NLP  
✔ Recommendation systems  

---

## 📐 Formula:

Cosine Similarity:

cos(θ) = (A · B) / (||A|| ||B||)

Cosine Distance:

1 - Cosine Similarity

---

## 🧮 Example:

A = (1, 0)  
B = (0, 1)

Dot product = 0  

Cosine similarity = 0  

Cosine distance = 1  

Means completely different direction.

---

# 📊 Quick Comparison Table

| Distance | Formula Type | Use Case |
|-----------|-------------|----------|
| Euclidean | Straight-line | K-Means |
| Manhattan | Grid-based | KNN |
| Minkowski | General form | Flexible |
| Chebyshev | Max difference | Chess moves |
| Cosine | Angle-based | Text similarity |

---

# 📌 When to Use Which?

✔ Euclidean → Default choice  
✔ Manhattan → When outliers exist  
✔ Cosine → Text / high-dimensional data  
✔ Chebyshev → Maximum movement matters  
✔ Minkowski → Generalized version  

---

# 🚀 Final Summary

Distance metrics measure similarity.

✔ Euclidean → L2 norm  
✔ Manhattan → L1 norm  
✔ Minkowski → General form  
✔ Chebyshev → Max difference  
✔ Cosine → Angle similarity  

Distance metric choice affects model performance.
