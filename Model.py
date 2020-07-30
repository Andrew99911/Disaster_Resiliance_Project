"""
Created on Thu Jul 30 18:06:25 2020
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# Import Relevant Data
data = pd.read_csv(r"C:\Users\Andrew\OneDrive\Documents\Disaster Project\NOAA_Precep_Data.csv")

# NEED DATA!!! ADJUST REST OF FILE LATER

# Take mean average for similar dates
data = data.groupby('DATE').mean().reset_index()
data.index = data['DATE']

#---- Label Encoder
for c in data.columns:
    if data[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

# Test train split
y = list(data['Target Var'])
X = data.drop('Target Var',axis=1)

# Drop Low Variance
sel = VarianceThreshold(threshold = 0.05)
sel.fit_transform(X)
sel_mask = sel.get_support()
X = X.loc[:,sel_mask]

# Drop High Correlations
corr_matrix = X.corr().abs()
corr_mask = np.triu(np.ones_like(corr_matrix,dtype = bool))
corr_matrix_tri = corr_matrix.mask(corr_mask)
to_drop = [c for c in corr_matrix_tri.columns if any(corr_matrix_tri[c] >0.95)]
X = X.drop(to_drop,axis = 1)

# Standardize X
std = StandardScaler()
X = std.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state = 12)
        
# Parameter List
parameters = {'objective':['reg:linear'],
              'learning_rate': [0.1],
              'n_estimators': [1000],
              'max_depth': [6],
              'colsample_bytree': [0.7],
              'gamma': [0]}

# Create XGBoost Model
gbm = xgb.XGBRegressor()

# Create XGB Grid Search
xgb_grid = GridSearchCV(gbm,parameters,cv = 5,scoring = 'r2')

# Fit the Model
eval_set = [(X_test,y_test)]
xgb_grid.fit(X_train,y_train,early_stopping_rounds = 50, eval_metric = 'logloss',eval_set = eval_set)

# Find best params
print('My Best Run: ', xgb_grid.best_params_)
model = xgb_grid.best_estimator_
model.fit(X_train,y_train)

# Predict Values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Print R^2 Values
print('Test R2 = ', r2_score(y_train,y_pred_train))
print('Test R2 = ', r2_score(y_test,y_pred_test))

# Print a Parity Plot
fig, ax = plt.subplots()
ax.scatter(y_train, y_pred_train, color = 'b', alpha = 0.2)
ax.scatter(y_train, y_pred_train, color = 'b', alpha = 0.2)
ax.plot([0,1],[0,1],'k--',lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
