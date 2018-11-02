"""

Usando DASK para abrir o Yellow Trip Data

"""
import os
from dask import dataframe as dd
from matplotlib import pyplot as plt

current_file = os.path.abspath(os.path.dirname(__file__))

# Read all .csv files: df
df = dd.read_csv(current_file + '\\yellow_tripdata_2015-*.csv',
                 assume_missing=True)

# Make column 'tip_fraction'
df['tip_fraction'] = df['tip_amount'] / (df['total_amount'] - df['tip_amount'])

# Convert 'tpep_dropoff_datetime' column to datetime objects
df['tpep_dropoff_datetime'] = dd.to_datetime(df['tpep_dropoff_datetime'])

# Construct column 'hour'
df['hour'] = df['tpep_dropoff_datetime'].dt.hour

# Filter rows where payment_type == 1: credit
credit = df.loc[df['payment_type'] == 1]

# Group by 'hour' column: hourly
hourly = credit.groupby('hour')

# Aggregate mean 'tip_fraction' and print its data type
result = hourly['tip_fraction'].mean()

# Perform the computation
tip_frac = result.compute()

# Generate a line plot using .plot.line()
tip_frac.plot.line()
plt.ylabel('Tip fraction')
plt.show()
