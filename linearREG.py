import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegression:
    def __init__(self):
        self.coefficient_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = 0
        denominator = 0

        for i in range(len(X)):
            numerator += (X[i] - x_mean) * (y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2

        self.coefficient_ = numerator / denominator
        self.intercept_ = y_mean - self.coefficient_ * x_mean

        return self.coefficient_, self.intercept_
    
    def predict(self, x):
        if self.coefficient_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before prediction.")

        return self.coefficient_ * x + self.intercept_


df = pd.read_csv("tvmarketing.csv")

x = df["TV"]
y = df["Sales"]

model = LinearRegression()

coef, inter = model.fit(x, y)

line = model.predict(x)

print("Prediction for TV = 5:", model.predict(5))
print("Coefficient:", coef)
print("Intercept:", inter)


plt.scatter(x, y, label="Real data")
plt.plot(x, line, "r", label="Regression line")
plt.xlabel("TV Marketing Budget")
plt.ylabel("Sales")
plt.legend()
plt.show()
