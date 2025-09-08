# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import X, X_encoded, X_scaled, y

# %%
# Fit Random Forest
rf= RandomForestClassifier(random_state=42)
rf.fit(X_encoded,y)

importances= pd.Series(rf.feature_importances_, index= X.columns)
importances= importances.sort_values(ascending=False)

plt.figure(figsize=(5,6))
importances.plot(kind='bar')
plt.title("Feature Importance from Random Forest")
plt.show()

estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=5)
rfe.fit(X_scaled, y)

selected_features = X.columns[rfe.support_]
print("Selected Features via RFE:", selected_features)

X_scaled_chi = MinMaxScaler().fit_transform(X_encoded)
chi_scores, p_values = chi2(X_scaled_chi, y)
chi_results = pd.Series(p_values, index=X.columns)
chi_results = chi_results.sort_values()
print("Chi-Square p-values:\n",chi_results)

rf_top = set(importances.index[:5])
chi_top = set(chi_results.index[:5])
rfe_top = set(selected_features)

confirmed_features = (rf_top & chi_top) | (rf_top & rfe_top) | (chi_top & rfe_top)

if len(confirmed_features) < 5:
    confirmed_features = rfe_top

top_features = list(confirmed_features)
X_selected = X[top_features]

print("\nSelected features confirmed by at least 2 methods:\n", X_selected)



