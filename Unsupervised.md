# 🤖 Unsupervised Learning – Complete Guide (With Models, Terms & Examples)

Unsupervised Learning is a type of Machine Learning where:

👉 There is NO target/output column  
👉 Model tries to find patterns in data by itself  

---

# 📌 Supervised vs Unsupervised

| Feature | Supervised | Unsupervised |
|----------|------------|--------------|
| Target Variable | Yes | No |
| Example | Spam Detection | Customer Segmentation |
| Output | Predict value | Find hidden patterns |

---

# 🎯 Where Unsupervised Learning is Used?

✔ Customer Segmentation  
✔ Market Basket Analysis  
✔ Anomaly Detection  
✔ Data Compression  
✔ Pattern Recognition  
✔ Recommendation Systems  

---

# 🔹 Types of Unsupervised Learning

1️⃣ Clustering  
2️⃣ Association Rule Learning  
3️⃣ Dimensionality Reduction  
4️⃣ Anomaly Detection  

---

# 🧩 1️⃣ Clustering

Clustering means grouping similar data points together.

Example:
Group customers based on:
- Age
- Income
- Spending score

---

## Important Terms in Clustering

### 🔹 Cluster
Group of similar data points.

### 🔹 Centroid
Center point of a cluster.

In K-Means:
Centroid = Mean of all points in that cluster.

### 🔹 Distance Metric
Used to measure similarity.

Common distances:
- Euclidean Distance
- Manhattan Distance

### 🔹 Inertia (WCSS)
Within Cluster Sum of Squares.
Measures how tightly data points are grouped.

Lower inertia = Better clustering.

---

# 📊 K-Means Clustering

Most popular clustering algorithm.

## How It Works:

1. Choose K (number of clusters)
2. Randomly initialize K centroids
3. Assign each point to nearest centroid
4. Recalculate centroids
5. Repeat until centroids stop changing

---

## Example Code

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0]])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.cluster_centers_)
print(kmeans.labels_)
```

---

## Choosing K (Elbow Method)

```python
inertia = []

for k in range(1, 10):
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertia.append(model.inertia_)
```

Plot inertia vs K → Choose elbow point.

---

# 📊 Hierarchical Clustering

Builds cluster tree (Dendrogram).

Two Types:
- Agglomerative (Bottom-Up)
- Divisive (Top-Down)

---

## Dendrogram

Tree-like diagram showing cluster merging.

```python
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X, method='ward')
dendrogram(linked)
```

---

# 📊 DBSCAN (Density-Based Clustering)

Density-Based Spatial Clustering.

Groups points that are close together.

Good for:
✔ Noise detection  
✔ Arbitrary shaped clusters  

Important Parameters:
- eps (distance radius)
- min_samples (minimum points)

---

## Example

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=3, min_samples=2)
model.fit(X)

print(model.labels_)
```

---

# 📉 2️⃣ Dimensionality Reduction

Used when dataset has too many features.

Goal:
Reduce features but keep important information.

---

## PCA (Principal Component Analysis)

Transforms data into fewer dimensions.

Important Terms:

### 🔹 Principal Component
New feature created from original features.

### 🔹 Variance
Amount of information retained.

---

## Example

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
```

---

# 📦 3️⃣ Association Rule Learning

Used in Market Basket Analysis.

Example:
People who buy bread also buy butter.

---

## Important Terms

### 🔹 Support
How frequently item appears.

### 🔹 Confidence
Probability of buying Y given X.

### 🔹 Lift
Strength of rule.
Lift > 1 means strong relationship.

---

## Apriori Algorithm

Used to generate association rules.

Example:

```python
from mlxtend.frequent_patterns import apriori
```

---

# 🚨 4️⃣ Anomaly Detection

Detect unusual data points.

Used in:
✔ Fraud detection  
✔ Network security  
✔ Fault detection  

Algorithms:
- Isolation Forest
- One-Class SVM

---

# 📌 Isolation Forest Example

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest()
model.fit(X)

predictions = model.predict(X)
```

Output:
1 → Normal  
-1 → Anomaly  

---

# 📊 Evaluation in Unsupervised Learning

Since no labels:

We use:

✔ Silhouette Score  
✔ Inertia  
✔ Davies-Bouldin Score  

---

## Silhouette Score

Range: -1 to 1

Higher = Better clustering

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, kmeans.labels_)
print(score)
```

---

# 📘 Important Terms Summary

| Term | Meaning |
|------|---------|
| Cluster | Group of similar data |
| Centroid | Center of cluster |
| Inertia | Compactness measure |
| Dendrogram | Tree of clusters |
| Principal Component | New reduced feature |
| Support | Frequency of item |
| Confidence | Conditional probability |
| Lift | Strength of rule |
| Outlier | Unusual data point |

---

# 🔥 When to Use Which Algorithm?

| Problem | Algorithm |
|----------|------------|
| Simple clustering | K-Means |
| Hierarchy needed | Hierarchical |
| Noise present | DBSCAN |
| Reduce features | PCA |
| Market basket | Apriori |
| Fraud detection | Isolation Forest |

---


# 🚀 Final Summary

✔ Unsupervised learning finds hidden patterns  
✔ No target variable  
✔ K-Means most common  
✔ PCA for dimensionality reduction  
✔ DBSCAN for noise handling  
✔ Association rules for shopping analysis  

Unsupervised Learning = Discover patterns without labels
