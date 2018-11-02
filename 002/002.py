# Import Lasso
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file into a DataFrame: df
df_columns = pd.Index(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality'])
columns=['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'life','child_mortality']
df = pd.read_csv('gapminder.csv', usecols =columns)

# Create arrays for features and target variable
y = df['life'].values
X = df.drop('life', axis=1).values

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
