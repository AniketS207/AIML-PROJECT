import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Energy_consumption.csv")

#print first 5 rows of the dataset
print(data.head())

#print last 5 rows of the dataset
print(data.tail())

#numbers of rows and columns in the dataset
print(data.shape)

#getting some info about the data
print(data.info())

#checking for missing values
data.isnull().sum()

#statistical measures of the data
data.describe()

#split the data into input features (X) and target variable(Y)
X = pd.get_dummies(data.drop('EnergyConsumption', axis=1))
y = data['EnergyConsumption']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)

# Input statements for additional features
timestamp_input = input("Enter the timestamp (e.g., '2022-01-02 05:00:00'): ")
temperature_input = float(input("Enter the temperature: "))
humidity_input = float(input("Enter the humidity: "))
squarefootage_input = float(input("Enter the square footage: "))
occupancy_input = int(input("Enter the occupancy: "))
HVACUsage_input = input("Enter the HVAC usage (e.g., 'On' or 'Off'): ")
LightingUsage_input = input("Enter the lighting usage (e.g., 'On' or 'Off'): ")
RenewableEnergy_input = float(input("Enter the renewable energy value: "))
day_of_week_input = input("Enter the day of the week (e.g., 'Tuesday'): ")
holiday_input = input("Is it a holiday? (Yes/No): ")

new_data = pd.DataFrame({
    'Timestamp': [timestamp_input],
    'Temperature': [temperature_input],
    'Humidity': [humidity_input],
    'SquareFootage': [squarefootage_input],
    'Occupancy': [occupancy_input],
    'HVACUsage': [HVACUsage_input],
    'LightingUsage': [LightingUsage_input],
    'RenewableEnergy': [RenewableEnergy_input],
    'DayOfWeek': [day_of_week_input],
    'Holiday': [holiday_input]
})

new_data_encoded = pd.get_dummies(new_data)

new_data_encoded_aligned = new_data_encoded.reindex(columns=X.columns, fill_value=0)

predicted_energy = model.predict(new_data_encoded_aligned)

print("Predicted Energy Consumption:", predicted_energy)