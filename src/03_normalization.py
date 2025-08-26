import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load Titanic dataset
train = pd.read_csv("data/train.csv")

# Normalize Age and Fare using Min-Max scaling
scaler = MinMaxScaler()
train[['Age_Norm', 'Fare_Norm']] = scaler.fit_transform(train[['Age', 'Fare']])

# Show first 10 rows
print("First 10 rows with normalized values:")
print(train[['Age', 'Age_Norm', 'Fare', 'Fare_Norm']].head(10))

# Save processed data
train.to_csv("data/train_normalized.csv", index=False)
