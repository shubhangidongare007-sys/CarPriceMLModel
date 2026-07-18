import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("========== Q5 ==========")
print("Model Trained Successfully")
print("Intercept:", model.intercept_)
print("Coefficients:")
print(model.coef_)