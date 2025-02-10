USE ROLE ACCOUNTADMIN;

-- Create warehouses
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH WITH WAREHOUSE_SIZE='X-SMALL';
CREATE WAREHOUSE IF NOT EXISTS FEATURE_STORE_WH WITH WAREHOUSE_SIZE='MEDIUM';

-- Create schema for setup
CREATE OR REPLACE SCHEMA SIMPLE_MLOPS_DEMO._DATA_GENERATION;

CREATE OR REPLACE FUNCTION SIMPLE_MLOPS_DEMO._DATA_GENERATION.GENERATE_TRANSACTIONS (REVENUE FLOAT, CHANNEL ARRAY)
  returns TABLE (CUSTOMER_ID INT, TRANSACTION_AMOUNT FLOAT, TRANSACTION_CHANNEL STRING)
  language python
  runtime_version = '3.11'
  packages=('numpy','scipy')
  handler = 'GenerateTransactions'
as
$$
from scipy.stats import truncnorm
import numpy as np

class GenerateTransactions:
    # Draw a number from a normal distribution with defined mean, std_dev, lower and upper bounds
    def get_norm_value(self, min, max, mean, std_dev):
        # Calculate the a and b parameters for truncnorm
        min = (min - mean) / std_dev
        max = (max - mean) / std_dev
        
        # Generate the truncated normal distribution
        truncated_data = truncnorm.rvs(min, max, loc=mean, scale=std_dev, size=1)[0]
        return truncated_data
        
    def process(self, revenue, in_shop_online):
        customer_id = 0
        while revenue > 0:
            # customer id
            if (customer_id >= 0) and (customer_id < 100):
                transaction_amount = np.round(self.get_norm_value(5, 25, 20, 5),2)
            elif (customer_id >= 100) and (customer_id < 200):
                transaction_amount = np.round(self.get_norm_value(25, 50, 40, 5),2)
            elif (customer_id >= 200) and (customer_id < 300):
                transaction_amount = np.round(self.get_norm_value(50, 100, 80, 10),2)
            else:
                transaction_amount = np.round(self.get_norm_value(100, 150, 120, 10),2)
            transaction_channel = np.random.choice(['IN_SHOP','ONLINE'],p=in_shop_online)
            revenue = revenue - transaction_amount
            if customer_id == 350:
                customer_id = 0
            customer_id = customer_id +1
            yield (customer_id, transaction_amount, transaction_channel) 
$$
;

-- Setup Procedure
CREATE OR REPLACE PROCEDURE SIMPLE_MLOPS_DEMO._DATA_GENERATION.DATA_GENERATION()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.11'
  PACKAGES = ('snowflake-snowpark-python','pandas','numpy')
  HANDLER = 'run'
  AS
$$
def run(session):
    from snowflake.snowpark.functions import lit, col
    import pandas as pd
    import numpy as np

    # Define the date range
    start_date = '2024-01-01'
    end_date = '2025-01-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define base revenue (this is a baseline you can adjust)
    base_revenue = 20000
    
    # Prepare a list to store computed records
    records = []
    
    for date in dates:
        # Determine the day-of-week: Monday=0, ... , Sunday=6
        weekday = date.weekday()
        
        # Weekday effect: Saturdays have the highest revenue,
        # Sundays are lower, and the rest are normal.
        if weekday == 5:       # Saturday
            weekday_factor = 1.5
        elif weekday == 6:     # Sunday
            weekday_factor = 0.9
        else:
            weekday_factor = 1.0
        
        # Month effect: June, July, August, and December get a boost.
        if date.month in [6, 7, 8, 12]:
            month_factor = 1.15
        else:
            month_factor = 1.0
        
        # Add random noise (mean 1, small standard deviation) to simulate natural fluctuations
        noise_factor = np.random.normal(loc=1, scale=0.05)
        
        # Compute the final revenue
        revenue = base_revenue * weekday_factor * month_factor * noise_factor
        revenue = max(0, revenue)  # Ensure no negative revenue
        
        # Append the result as a tuple (DATE, REVENUE)
        records.append((date, revenue))
    
    # Create a DataFrame from the records
    revenue_df = pd.DataFrame(records, columns=['DATE', 'REVENUE'])
    revenue_df = session.create_dataframe(revenue_df)
    revenue_in_shop = revenue_df.filter(col('DATE') < lit('2024-06-01'))
    revenue_online = revenue_df.filter(col('DATE') >= lit('2024-06-01'))
    
    revenue_in_shop.join_table_function('SIMPLE_MLOPS_DEMO._DATA_GENERATION.GENERATE_TRANSACTIONS', col('REVENUE'), lit([0.75,0.25])).drop('REVENUE').write.save_as_table(table_name='SIMPLE_MLOPS_DEMO._DATA_GENERATION._TRANSACTIONS', mode='overwrite')
    revenue_online.join_table_function('SIMPLE_MLOPS_DEMO._DATA_GENERATION.GENERATE_TRANSACTIONS',col('REVENUE'), lit([0.25,0.75])).drop('REVENUE').write.save_as_table(table_name='SIMPLE_MLOPS_DEMO._DATA_GENERATION._TRANSACTIONS', mode='append')
    session.table('SIMPLE_MLOPS_DEMO._DATA_GENERATION._TRANSACTIONS').select('CUSTOMER_ID').distinct().write.save_as_table('SIMPLE_MLOPS_DEMO._DATA_GENERATION._CUSTOMERS')
    return "Demo Environment is setup."
$$
;
CALL SIMPLE_MLOPS_DEMO._DATA_GENERATION.DATA_GENERATION();

CREATE OR REPLACE NOTEBOOK SIMPLE_MLOPS_DEMO.PUBLIC.SIMPLE_MLOPS_DEMO_NOTEBOOK 
FROM '@SIMPLE_MLOPS_DEMO.PUBLIC.GITHUB_REPOSITORY_SNOWFLAKE_SIMPLE_MLOPS/branches/main/src/' 
MAIN_FILE = 'SIMPLE_MLOPS_DEMO_NOTEBOOK.ipynb' 
QUERY_WAREHOUSE = COMPUTE_WH
IDLE_AUTO_SHUTDOWN_TIME_SECONDS=3600;
ALTER NOTEBOOK SIMPLE_MLOPS_DEMO.PUBLIC.SIMPLE_MLOPS_DEMO_NOTEBOOK  ADD LIVE VERSION FROM LAST;