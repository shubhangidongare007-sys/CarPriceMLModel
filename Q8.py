import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Independent and Dependent Variables
X = df.drop("Price", axis=1)
y = df["Price"]

# One Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Save Column Names
columns = X.columns

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("LR_car.pkl", "wb"))

# Save Scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save Columns
pickle.dump(columns, open("columns.pkl", "wb"))

print("========== Q8 ==========")
print("Model Saved Successfully")
print("Scaler Saved Successfully")
print("Columns Saved Successfully")