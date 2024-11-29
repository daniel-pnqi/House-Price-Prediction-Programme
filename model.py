from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split

#loading up my dataset
file_path = 'C:\\Users\\danie\\Documents\\datasets\\house_dataset.csv'
data = pd.read_csv(file_path)

#dropping all futile columns. dataset was very uncleaned
columns_to_drop = [
    'tenure', 'fullAddress','outcode','propertyType', 'postcode', 'country', 
    'currentEnergyRating', 'saleEstimate_confidenceLevel', 'saleEstimate_ingestedAt', 
    'saleEstimate_valueChange.numericChange', 'saleEstimate_valueChange.percentageChange', 
    'saleEstimate_valueChange.saleDate', 'history_date', 'history_percentageChange', 
    'history_numericChange', 'saleEstimate_upperPrice', 'saleEstimate_lowerPrice', 'rentEstimate_currentPrice',
    'rentEstimate_lowerPrice', 'rentEstimate_upperPrice', 'longitude', 'latitude', 'history_price'
]
data = data.drop(columns=columns_to_drop, axis=1)

#handling missing values as again, dataset was very unclean
imputer = SimpleImputer(strategy='median')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

#features and target
X = data.drop('saleEstimate_currentPrice', axis=1,)  
y = data['saleEstimate_currentPrice']              

print(X)

#feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#log transformation of the target variable
y_log = np.log1p(y)

#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

#training Ridge model
model = Ridge(alpha=1.0)
print("X_train:\n", X_train)
model.fit(X_train, y_train)

from sklearn.preprocessing import StandardScaler
import joblib

#scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform your features

#save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")


#predicting on the test set
y_pred = model.predict(X_test)

#evaluating the model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'house_price_model.pkl')



