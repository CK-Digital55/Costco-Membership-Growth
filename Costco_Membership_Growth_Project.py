
# Costco Global Membership Growth Prediction

## Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Data Preparation
# Load the datasets
# Assuming the datasets are named 'Paid_Memberships.csv' and 'Total_Memberships.csv'
paid_data = pd.read_csv("Paid_Memberships.csv")
total_data = pd.read_csv("Total_Memberships.csv")

# Merge the datasets on 'Year'
membership_data = pd.merge(paid_data, total_data, on="Year")

## Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 6))
sns.lineplot(x='Year', y='Paid_Memberships_Millions', data=membership_data, marker='o')
plt.title('Growth of Paid Memberships Worldwide (in millions)')
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='Year', y='Total_Memberships_Millions', data=membership_data, marker='o')
plt.title('Growth of Total Memberships Worldwide (in millions)')
plt.show()

## Model Building
# Prepare features and labels
X = membership_data[['Year']]
y_paid = membership_data['Paid_Memberships_Millions']
y_total = membership_data['Total_Memberships_Millions']

# Initialize the models
model_paid = LinearRegression()
model_total = LinearRegression()

# Train the models
model_paid.fit(X, y_paid)
model_total.fit(X, y_total)

## Evaluation
# Make predictions
y_paid_pred = model_paid.predict(X)
y_total_pred = model_total.predict(X)

# Calculate metrics
mse_paid = mean_squared_error(y_paid, y_paid_pred)
r2_paid = r2_score(y_paid, y_paid_pred)

mse_total = mean_squared_error(y_total, y_total_pred)
r2_total = r2_score(y_total, y_total_pred)

print(f"For Paid Memberships: R2 = {r2_paid}, MSE = {mse_paid}")
print(f"For Total Memberships: R2 = {r2_total}, MSE = {mse_total}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Year', y='Paid_Memberships_Millions', data=membership_data, label='Actual', color='blue')
sns.lineplot(x=membership_data['Year'], y=y_paid_pred, label='Predicted', color='red')
plt.title('Paid Memberships: Actual vs Predicted')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Year', y='Total_Memberships_Millions', data=membership_data, label='Actual', color='blue')
sns.lineplot(x=membership_data['Year'], y=y_total_pred, label='Predicted', color='red')
plt.title('Total Memberships: Actual vs Predicted')
plt.show()

## Future Predictions
# Create an array for the future years
future_years = np.array(range(2023, 2034)).reshape(-1, 1)

# Make predictions for future years
future_paid_memberships = model_paid.predict(future_years)
future_total_memberships = model_total.predict(future_years)

# Display the predictions
future_predictions = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted_Paid_Memberships_Millions': future_paid_memberships,
    'Predicted_Total_Memberships_Millions': future_total_memberships
})

print(future_predictions)

plt.figure(figsize=(8, 6))
sns.lineplot(x='Year', y='Predicted_Paid_Memberships_Millions', data=future_predictions, marker='o', color='green')
plt.title('Predicted Growth of Paid Memberships Worldwide (2023-2033)')
plt.show()

plt.figure(figsize=(8, 6))
sns.lineplot(x='Year', y='Predicted_Total_Memberships_Millions', data=future_predictions, marker='o', color='purple')
plt.title('Predicted Growth of Total Memberships Worldwide (2023-2033)')
plt.show()
