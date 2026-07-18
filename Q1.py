import pandas as pd

# Load Dataset
df = pd.read_csv("car_price_dataset.csv")

# Independent Features (X)
X = df.drop("Price", axis=1)

# Dependent Feature (Y)
Y = df["Price"]

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

print("\nIndependent Features:")
print(X.head())

print("\nDependent Feature:")
print(Y.head())