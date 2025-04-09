import pandas as pd
from sklearn.datasets import make_classification

# Generate the dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Save dataset to CSV
data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
data["Label"] = y

# Create a "data" directory if it doesn't exist
import os
if not os.path.exists("data"):
    os.makedirs("data")

# Save the dataset in the "data" folder
data.to_csv("data/dataset.csv", index=False)

print("Dataset generated and saved to data/dataset.csv")
