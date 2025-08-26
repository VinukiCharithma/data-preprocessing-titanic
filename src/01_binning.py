import pandas as pd

# Load Titanic dataset
train = pd.read_csv("data/train.csv")

# Create bins for Age
bins = [0, 12, 18, 60, 80]
labels = ['Child', 'Teen', 'Adult', 'Senior']
train['Age_Bin'] = pd.cut(train['Age'], bins=bins, labels=labels)

# Show first 10 rows
print("First 10 rows with Age bins:")
print(train[['Age', 'Age_Bin']].head(10))

# Save processed data
train.to_csv("data/train_binned.csv", index=False)
