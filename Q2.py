import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# ==========================
# Q2 - One Hot Encoding
# ==========================

print("\n========== Q2 ==========")

X = df.drop("Price", axis=1)
y = df["Price"]

X = pd.get_dummies(X, drop_first=True)

print("\nEncoded Dataset (First 5 Rows):")
print(X.head())

# ==========================
# Q3 - Feature Scaling
# ==========================

print("\n========== Q3 ==========")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Data (First 5 Rows):")
print(X_scaled[:5])

# ==========================
# Q4 - Train Test Split
# ==========================

print("\n========== Q4 ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

print("\nX_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape:", y_test.shape)