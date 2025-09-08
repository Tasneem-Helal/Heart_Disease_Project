# %%
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


# %%
###Data Preprocessing & Cleaning###

# Always work from the project root
project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
os.chdir(project_root)

# Make sure data folder exists
os.makedirs('data', exist_ok=True)

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# Make X a full copy to avoid warnings
X = X.copy()

# separate numeric and categorigal columns 
numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(exclude=['number']).columns

# fill numeric columns with median
for col in numeric_cols:
    X[col]= X[col].fillna(X[col].median())

# fill categorigal columns with mode
for col in categorical_cols:
    X[col]= X[col].fillna(X[col].mode()[0])

# for the target 
y= y.fillna(y.mode().iloc[0]).values.ravel()

# Perform data encoding
X_encoded = pd.get_dummies(X)

# Standardize numerical features
scaler= StandardScaler()
X_scaled= scaler.fit_transform(X_encoded)

print(f"Original:\n{X[:5].values}")
print(f"\nScaled:\n{X_scaled[:5]}")

# Save cleaned dataset locally if you want

pd.DataFrame(X_encoded).to_csv('data/heart_disease_clean.csv', index=False)




