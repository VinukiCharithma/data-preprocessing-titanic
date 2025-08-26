import pandas as pd
import numpy as np

# Load Titanic dataset
train = pd.read_csv("data/train.csv")

# Apply log transformation to Fare (add 1 to avoid log(0))
train['Fare_Log'] = np.log1p(train['Fare'])

# Show first 10 rows
print("First 10 rows with log-transformed Fare:")
print(train[['Fare', 'Fare_Log']].head(10))

# Save processed data
train.to_csv("data/train_transformed.csv", index=False)
