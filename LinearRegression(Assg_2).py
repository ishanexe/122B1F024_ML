import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Salary_Data.csv")

X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

B0 = model.intercept_
B1 = model.coef_[0]

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Intercept (B0): {B0:.2f}")
print(f"Slope (B1): {B1:.2f}")
print(f"RMSE: {rmse:.3f}")

plt.scatter(X_test, y_test, color="blue", label="Actual Salary")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
