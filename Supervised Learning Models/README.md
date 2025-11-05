
## 1. **Linear Regression**

### 📘 Description :-

Linear Regression is a supervised learning algorithm that models the relationship between input features and a continuous target variable by fitting a best-fit straight line. It’s primarily used when data shows a linear correlation between independent and dependent variables.


### ⚙️ **Key Points**

* **Type:** Regression
* **Output Type:** Continuous
* **Algorithm / Technique:** Ordinary Least Squares (OLS)
* **Best Use Case:** Predicting numeric outcomes such as prices, revenue, or growth trends.


### 💻 **Code Example**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression().fit(X_train, y_train)
print("R² Score:", model.score(X_test, y_test))
```

👉 [Try it on Colab](https://colab.research.google.com/)


###  **📊 Diagram / Workflow Prompt**

<img src="https://github.com/codeandcrush089/ML-Models-Visualized/blob/main/Img/Linear%20Regression%201.JPG" width="600">

### 📈 **Pros & Cons**

* ✅ Simple and highly interpretable
* ✅ Fast and efficient for small to medium datasets
* ⚠️ Performs poorly with non-linear or highly correlated features


### 📚 **Real-World Use Case**

🏠 Used in **house price prediction**, **sales forecasting**, and **trend analysis** where relationships between variables are approximately linear.


---

## 2 **Ridge Regression**

### 📘 Description

Ridge Regression is an advanced form of Linear Regression that adds **L2 regularization** to penalize large coefficients. It’s used to prevent overfitting and handle multicollinearity in regression models.


### ⚙️ **Key Points**

* **Type:** Regression
* **Output Type:** Continuous
* **Algorithm / Technique:** L2 Regularization (Tikhonov Regularization)
* **Best Use Case:** When features are highly correlated or when overfitting is observed in linear models.


### 💻 **Code Example**

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print("R² Score:", model.score(X_test, y_test))
```

👉 [Try it on Colab](https://colab.research.google.com/)


### 📊 **Diagram / Workflow Prompt**

<img src="https://github.com/codeandcrush089/ML-Models-Visualized/blob/main/Img/Ridge%20Regression%201.JPG" width="600">


### 📈 **Pros & Cons**

* ✅ Reduces overfitting by shrinking coefficients
* ✅ Works well with multicollinearity
* ⚠️ Doesn’t perform automatic feature selection (all coefficients remain non-zero)


### 📚 **Real-World Use Case**

📊 Used in **financial forecasting**, **healthcare cost prediction**, and **energy demand estimation** where features are interrelated.


---

## 3 **Lasso Regression**

### 📘 Description

Lasso Regression adds **L1 regularization** to the Linear Regression model, penalizing the absolute value of coefficients. It’s mainly used for **feature selection** by driving irrelevant feature coefficients to zero.


### ⚙️ **Key Points**

* **Type:** Regression
* **Output Type:** Continuous
* **Algorithm / Technique:** L1 Regularization
* **Best Use Case:** When you need both prediction and automatic feature selection.


### 💻 **Code Example**

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
print("R² Score:", model.score(X_test, y_test))
```

👉 [Try it on Colab](https://colab.research.google.com/)


### 📊 **Diagram / Workflow Prompt**

<img src="https://github.com/codeandcrush089/ML-Models-Visualized/blob/main/Img/Lasso%20Regression.JPG" width="600">


### 📈 **Pros & Cons**

* ✅ Performs feature selection by setting coefficients to zero
* ✅ Prevents overfitting and improves model generalization
* ⚠️ May remove useful correlated features unintentionally


### 📚 **Real-World Use Case**

💡 Used in **genetic data analysis**, **marketing mix modeling**, and **sparse signal recovery** where only a few features drive predictions.

---

## 4 **Polynomial Regression**

### 📘 Description

Polynomial Regression extends Linear Regression by modeling the relationship between input and target variables as an **nth-degree polynomial**. It’s useful when the data shows **non-linear trends** that a straight line can’t capture.


### ⚙️ **Key Points**

* **Type:** Regression
* **Output Type:** Continuous
* **Algorithm / Technique:** Polynomial Feature Transformation + Linear Regression
* **Best Use Case:** Modeling **non-linear relationships** between independent and dependent variables.


### 💻 **Code Example**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
print("R² Score:", model.score(X_poly, y))
```

👉 [Try it on Colab](https://colab.research.google.com/)


### 📊 **Diagram / Workflow Prompt**

<img src="https://github.com/codeandcrush089/ML-Models-Visualized/blob/main/Img/Polynomial%20Regression%201.JPG" width="600">

### 📈 **Pros & Cons**

* ✅ Captures non-linear relationships effectively
* ✅ Simple to implement and interpret up to moderate degrees
* ⚠️ High-degree polynomials may cause **overfitting** and poor generalization


### 📚 **Real-World Use Case**

📈 Used in **growth curve modeling**, **temperature trend analysis**, and **demand forecasting** where patterns are **non-linear but continuous**.

---


