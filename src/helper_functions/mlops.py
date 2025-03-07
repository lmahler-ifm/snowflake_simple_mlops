
# Snowflake Snowpark imports
from snowflake.snowpark import Session 
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import lit, col
from snowflake.ml.registry import Registry
from snowflake.ml.modeling.metrics import mean_absolute_percentage_error
from snowflake.ml.feature_store import (
    FeatureStore,
    CreationMode
)
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.registry import Registry

# Third-party imports
import pandas as pd
import time
from datetime import datetime
import calendar
from tabulate import tabulate
from datetime import timedelta
import json


def is_feature_store_updated(session: Session, timestamp_df):
    """
    Checks whether all feature store tables have been refreshed after the given timestamp.
    
    Parameters:
        session (Session): The Snowpark session object.
        timestamp_df: A Snowpark DataFrame containing a 'TIMESTAMP' column, representing the reference time.
    
    Returns:
        True if all feature store tables have been updated after the reference timestamp; otherwise, False.
    """
    # Query the dynamic table refresh history for the Feature Store.
    feature_store_refreshes = (
        session.table_function('INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY')
        # Filter to include only tables that belong to the Feature Store.
        .filter(col('QUALIFIED_NAME').startswith(f'{session.get_current_database()}.FEATURE_STORE.'))
        # Select relevant columns: table name, state, and the refresh end time.
        .select('NAME', 'STATE', 'REFRESH_END_TIME')
        # Order the records by the refresh end time in descending order.
        .order_by(col('REFRESH_END_TIME').desc())
        # Group by table name and state, then pick the latest refresh end time.
        .group_by('NAME', 'STATE')
        .agg(F.max('REFRESH_END_TIME').as_('REFRESH_END_TIME'))
        # Calculate the number of seconds since each table was last refreshed.
        .with_column('SECONDS_SINCE_LAST_REFRESH', F.datediff('second', col('REFRESH_END_TIME'), F.current_timestamp()))
        # Cross join with the provided timestamp dataframe to attach the reference timestamp to each row.
        .cross_join(timestamp_df)
        # Create an 'UPDATED' column that is True if the table's last refresh is later than the reference timestamp.
        .with_column('UPDATED', col('REFRESH_END_TIME') > col('TIMESTAMP'))
    )

    # If any table in the feature store has not been updated (i.e., UPDATED == False), return False.
    if feature_store_refreshes.filter(col('UPDATED') == False).count() > 0:
        return False
    else:
        return True


def wait_until_feature_store_updated(session: Session, interval=3):
    """
    Waits until the Feature Store has been updated relative to the current timestamp.
    
    Parameters:
        session (Session): The Snowpark session object.
        interval (int): Number of seconds to wait between checks. Defaults to 3 seconds.
    """
    # Create a one-row dataframe containing the current timestamp.
    # session.range(1) creates a dataframe with one row and an 'ID' column, which we drop.
    ts = session.range(1) \
                .with_column('TIMESTAMP', F.current_timestamp()) \
                .drop('ID') \
                .cache_result()
    # Record the starting time for logging purposes.
    start_time = time.time()

    # Continuously check if the feature store has been updated.
    while not is_feature_store_updated(session, ts):
        # Print a dynamic status message with elapsed time (overwriting the same line).
        print(f"\rWaiting for updated Feature Store ... ({int(time.time() - start_time)} seconds.)", end="", flush=True)
        # Pause for the specified interval before checking again.
        time.sleep(interval)


def last_day_of_month(year: int, month: int) -> datetime:
    """
    Returns the last day of a given month and year.
    
    Parameters:
        year (int): The year.
        month (int): The month.
    
    Returns:
        A datetime object representing the last day of the specified month.
    """
    # 'calendar.monthrange' returns a tuple: (weekday of first day, number of days in month).
    last_day = calendar.monthrange(year, month)[1]
    # Return a datetime object for the last day of the given month and year.
    return datetime(year, month, last_day)


def generate_last_days(start_date: str, end_date: str):
    """
    Generates a list of the last day (as a string in 'YYYY-MM-DD' format) of each month within the given date range.
    
    Parameters:
        start_date (str): The start date in "YYYY-MM-DD" format.
        end_date (str): The end date in "YYYY-MM-DD" format.
    
    Returns:
        A list of strings, each representing the last day of a month within the specified range.
    """
    # Convert the input date strings to datetime objects.
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    results = []
    # Initialize 'current' as the last day of the start month.
    current = last_day_of_month(start.year, start.month)
    
    # Loop until the current date exceeds the end date.
    while current <= end:
        # Append the current date formatted as 'YYYY-MM-DD'.
        results.append(current.strftime("%Y-%m-%d"))
        
        # Determine the next month and year.
        next_month = current.month + 1
        next_year = current.year + (1 if next_month > 12 else 0)
        # If next_month exceeds 12, reset it to January.
        next_month = next_month if next_month <= 12 else 1
        
        # Set 'current' to the last day of the next month.
        current = last_day_of_month(next_year, next_month)
    
    return results


def get_feature_df(session, feature_cutoff_date):
    """
    Retrieves the feature dataframe for customers as of a given feature cutoff date.
    
    Parameters:
        session (Session): The Snowpark session object.
        feature_cutoff_date (str): The cutoff date for features (in a format accepted by F.to_date).
    
    Returns:
        A Snowpark DataFrame with customer features and a placeholder column for NEXT_MONTH_REVENUE.
    """
    # Initialize the Feature Store.
    fs = FeatureStore(
        session=session, 
        database=session.get_current_database(), 
        name='FEATURE_STORE', 
        default_warehouse=session.get_current_warehouse(),
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    
    # Retrieve all feature views (version 'V1') from the Feature Store.
    fvs = [
        fs.get_feature_view(n[0], 'V1') 
        for n in fs.list_feature_views().select('NAME').to_pandas().values
    ]
    
    # Create a base (spine) DataFrame containing distinct CUSTOMER_IDs and the feature cutoff date.
    feature_df = session.table(f'{session.get_current_database()}.RETAIL_DATA.CUSTOMERS') \
                        .select('CUSTOMER_ID') \
                        .distinct() \
                        .with_column('FEATURE_CUTOFF_DATE', F.to_date(lit(feature_cutoff_date)))
    
    # Retrieve feature values from the Feature Store for the specified cutoff date.
    feature_df = fs.retrieve_feature_values(
        spine_df=feature_df,
        features=fvs,
        spine_timestamp_col="FEATURE_CUTOFF_DATE"
    )
    
    # Add a placeholder column for NEXT_MONTH_REVENUE (set to NULL, cast to the desired numeric type).
    feature_df = feature_df.with_column('NEXT_MONTH_REVENUE', lit(None).cast('number(38,2)'))
    
    return feature_df

def simulate_model_performance(session, start_date, end_date, model_version, generate_data=False):
    # ------------------------------------------------------------------------------
    # Optionally Generate New Transaction Data
    # ------------------------------------------------------------------------------
    if generate_data:
        # Filter new transactions from the data generation table for the specified date range.
        new_transactions = session.table(f'{session.get_current_database()}._DATA_GENERATION._TRANSACTIONS') \
                                  .filter(col('DATE').between(start_date, end_date))
        # Append the new transactions to the main transactions table.
        new_transactions.write.save_as_table(
            table_name=f'{session.get_current_database()}.RETAIL_DATA.TRANSACTIONS', 
            mode='append'
        )
        
        # Wait for the Feature Store to update with the new data.
        wait_until_feature_store_updated(session)
        print('')  # Optional: add a blank line for readability.
    
    # ------------------------------------------------------------------------------
    # Retrieve the Registered Model from the Model Registry
    # ------------------------------------------------------------------------------
    reg = Registry(
        session=session, 
        database_name=session.get_current_database(), 
        schema_name='MODEL_REGISTRY', 
        options={'enable_monitoring': True},
    )
    # Retrieve the specific model version from the registry.
    registered_model = reg.get_model('CUSTOMER_REVENUE_MODEL').version(model_version)

    # Retrieve the source table of the model version's model monitor
    model_monitors = pd.DataFrame(reg.show_model_monitors())
    model_monitor = model_monitors[model_monitors['name'] == reg.get_monitor(model_version=registered_model).name]
    model_monitor_source = json.loads(model_monitor['source'].iloc[0])
    model_monitor_source = f"{model_monitor_source['database_name']}.{model_monitor_source['schema_name']}.{model_monitor_source['name']}"
    
    # ------------------------------------------------------------------------------
    # Generate Predictions for Each Day in the Date Range
    # ------------------------------------------------------------------------------
    # Loop over each date generated by the helper function
    # subtract 1 day to make sure prior month is covered
    for date in generate_last_days(start_date, end_date):
        print(f'Generated predictions with features until: {date}.')
        # Retrieve the feature dataframe using the current date as the cutoff.
        feature_df = get_feature_df(session, feature_cutoff_date=date)
    
        # Generate predictions using the registered model.
        predictions = registered_model.run(feature_df, function_name='PREDICT')
        # Ensure the FEATURE_CUTOFF_DATE column is of type timestamp.
        predictions = predictions.with_column('FEATURE_CUTOFF_DATE', F.col('FEATURE_CUTOFF_DATE').cast('timestamp'))
        # Cast the prediction column to the appropriate numeric type.
        predictions = predictions.with_column('NEXT_MONTH_REVENUE_PREDICTION', F.col('NEXT_MONTH_REVENUE_PREDICTION').cast('number(38,2)'))
        # Append the predictions to the source table used for model monitoring.
        predictions.write.save_as_table(
            table_name=model_monitor_source, 
            mode='append'
        )
    
    # ------------------------------------------------------------------------------
    # Prepare Actual Revenue Values for the Given Date Range
    # ------------------------------------------------------------------------------
    # Load transaction data for the specified date range.
    actual_values_df = (
        session.table(f'{session.get_current_database()}.RETAIL_DATA.TRANSACTIONS')
        .filter(col('DATE').between(start_date, end_date))
        # Extract year and month from the transaction date.
        .with_column('YEAR', F.year(col('DATE')))
        .with_column('MONTH', F.month(col('DATE')))
        # Aggregate revenue by CUSTOMER_ID, YEAR, and MONTH.
        .group_by(['CUSTOMER_ID', 'YEAR', 'MONTH'])
        .agg(F.sum('TRANSACTION_AMOUNT').cast('decimal(38,2)').as_('REVENUE'))
        # Determine the last day of each month by computing the date from parts and subtracting one day.
        .with_column('DATE', F.date_add(F.date_from_parts(col('YEAR'), col('MONTH'), lit(1)), lit(-1)).cast('timestamp'))
        # Remove helper columns as they are no longer needed.
        .drop(['YEAR', 'MONTH'])
    )
    
    # ------------------------------------------------------------------------------
    # Ensure All Customers are Represented in Actual Values
    # ------------------------------------------------------------------------------
    # Retrieve a distinct list of all customers.
    customers_df = session.table(f'{session.get_current_database()}.RETAIL_DATA.CUSTOMERS').select('CUSTOMER_ID').distinct()
    
    # Join actual revenue values with the full customer list (outer join) to include customers without transactions.
    actual_values_df = actual_values_df.join(customers_df, on=['CUSTOMER_ID'], how='outer')
    # Fill missing revenue values with 0 for customers with no transactions.
    actual_values_df = actual_values_df.fillna(0, subset='REVENUE')
    
    # ------------------------------------------------------------------------------
    # Update the Model Monitor Source Table with Actual Revenue Values
    # ------------------------------------------------------------------------------
    # Load the source table that holds the prediction data for model monitoring.
    source_table = session.table(model_monitor_source)
    
    # Update the source table by matching records on FEATURE_CUTOFF_DATE and CUSTOMER_ID,
    # assigning the actual revenue values to the NEXT_MONTH_REVENUE column.
    update_result = source_table.update(
        condition=(
            (source_table['FEATURE_CUTOFF_DATE'] == actual_values_df['DATE']) &
            (source_table['CUSTOMER_ID'] == actual_values_df['CUSTOMER_ID'])
        ),
        assignments={
            "NEXT_MONTH_REVENUE": actual_values_df['REVENUE'],
        },
        source=actual_values_df
    )
    
    # Print the number of rows that were updated.
    print(f'Updated {update_result.rows_updated} rows in source table of model monitor.')
    
    return


def evaluate_against_production_model(session, new_model_version, test_df):
    # ------------------------------------------------------------------------------
    # Initialize the Registry for Model Management
    # ------------------------------------------------------------------------------
    # Create a Registry instance with monitoring enabled.
    reg = Registry(
        session=session, 
        database_name=session.get_current_database(), 
        schema_name='MODEL_REGISTRY', 
        options={'enable_monitoring': True},
    )
    
    # ------------------------------------------------------------------------------
    # Evaluate the Current Production Model
    # ------------------------------------------------------------------------------
    # Retrieve the production model for 'CUSTOMER_REVENUE_MODEL' using the 'PRODUCTION' alias.
    production_model = reg.get_model('CUSTOMER_REVENUE_MODEL').version('PRODUCTION')
    
    # Run predictions on the test dataset using the production model.
    production_model_predictions = production_model.run(test_df, function_name='PREDICT')
    
    # Calculate the Mean Absolute Percentage Error (MAPE) for the production model predictions.
    production_model_mape = mean_absolute_percentage_error(
        df=production_model_predictions, 
        y_true_col_names="NEXT_MONTH_REVENUE", 
        y_pred_col_names="NEXT_MONTH_REVENUE_PREDICTION"
    )
    
    # ------------------------------------------------------------------------------
    # Evaluate the New Development Model
    # ------------------------------------------------------------------------------
    # Retrieve the new development model using its version number.
    development_model = reg.get_model('CUSTOMER_REVENUE_MODEL').version(new_model_version)
    
    # Run predictions on the test dataset using the development model.
    development_model_predictions = development_model.run(test_df, function_name='PREDICT')
    
    # Calculate the MAPE for the development model predictions.
    development_model_mape = mean_absolute_percentage_error(
        df=development_model_predictions, 
        y_true_col_names="NEXT_MONTH_REVENUE", 
        y_pred_col_names="NEXT_MONTH_REVENUE_PREDICTION"
    )
    
    # Print both MAPE values for comparison.
    print(development_model_mape, production_model_mape)
    
    # ------------------------------------------------------------------------------
    # Compare and Update Model Aliases Based on Performance
    # ------------------------------------------------------------------------------
    if development_model_mape < production_model_mape:
        print(f"New model with version {new_model_version} has a lower MAPE compared to current production model.")
        
        # Create a DataFrame to display the MAPE values of both models.
        mape_values_df = pd.DataFrame(
            [
                ['PRODUCTION', production_model_mape],
                [new_model_version, development_model_mape]
            ],
            columns=['VERSION', 'MAPE']
        )
        
        # Display the MAPE values in a formatted table.
        print(tabulate(mape_values_df, headers='keys', tablefmt='grid'))
        
        # Update model aliases:
        # - Remove the 'PRODUCTION' alias from the current production model.
        # - Set its alias to 'DEPRECATED'.
        # - Assign the 'PRODUCTION' alias to the new development model.
        production_model.unset_alias('PRODUCTION')
        production_model.set_alias('DEPRECATED')
        development_model.set_alias('PRODUCTION')
    else:
        print(f"Existing production model has a lower MAPE compared to the developed model.")

def train_new_model(session, feature_cutoff_date, target_start_date, target_end_date, model_version):
    # ------------------------------------------------------------------------------
    # Initialize Feature Store
    # ------------------------------------------------------------------------------
    # Create a FeatureStore instance using the current session, database, and warehouse.
    fs = FeatureStore(
        session=session, 
        database=session.get_current_database(), 
        name='FEATURE_STORE', 
        default_warehouse=session.get_current_warehouse(),
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    
    # Retrieve all feature views from the FeatureStore (using version 'V1' for each).
    fvs = [
        fs.get_feature_view(n[0], 'V1') 
        for n in fs.list_feature_views().select('NAME').to_pandas().values
    ]
    
    # ------------------------------------------------------------------------------
    # Create Training Dataset
    # ------------------------------------------------------------------------------
    print('Creating training dataset...')
    
    # Load the transactions data from the target table.
    target_df = session.table(f'{session.get_current_database()}.RETAIL_DATA.TRANSACTIONS')
    
    # Filter transactions between target_start_date and target_end_date.
    # Group by CUSTOMER_ID and aggregate the transaction amounts to compute NEXT_MONTH_REVENUE.
    # Add a FEATURE_CUTOFF_DATE column with the specified cutoff date.
    target_df = (
        target_df.filter(col('DATE').between(target_start_date, target_end_date))
                 .group_by('CUSTOMER_ID')
                 .agg(F.sum('TRANSACTION_AMOUNT').as_('NEXT_MONTH_REVENUE'))
                 .with_column('FEATURE_CUTOFF_DATE', F.to_date(lit(feature_cutoff_date)))
    )
    
    # Load the distinct customer IDs from the customers table.
    customers_df = session.table(f'{session.get_current_database()}.RETAIL_DATA.CUSTOMERS').select('CUSTOMER_ID').distinct()
    
    # Join the target data with the customer "spine" using an outer join to ensure all customers are included.
    spine_df = target_df.join(customers_df, on=['CUSTOMER_ID'], how='outer')
    
    # Fill missing revenue values with 0.
    spine_df = spine_df.fillna(0, subset='NEXT_MONTH_REVENUE')
    
    # Generate the training dataset using the FeatureStore.
    train_dataset = fs.generate_dataset(
        name=f"{session.get_current_database()}.FEATURE_STORE.NEXT_MONTH_REVENUE_DATASET",
        spine_df=spine_df,
        features=fvs,
        version=model_version,
        spine_timestamp_col="FEATURE_CUTOFF_DATE",
        spine_label_cols=["NEXT_MONTH_REVENUE"],
        include_feature_view_timestamp_col=False,
        desc="Training Dataset from September 2024"
    )
    
    # Convert the generated dataset to a Snowpark DataFrame.
    df = train_dataset.read.to_snowpark_dataframe()
    print('Training dataset created.')
    
    # ------------------------------------------------------------------------------
    # Train a New Model using XGBoost
    # ------------------------------------------------------------------------------
    print(f'Training new model with version {model_version}...')
    
    # Split the data into training (90%) and testing (10%) sets.
    train_df, test_df = df.random_split(weights=[0.9, 0.1], seed=0)
    
    # Identify feature columns by dropping identifier and label columns.
    feature_columns = train_df.drop(['CUSTOMER_ID', 'FEATURE_CUTOFF_DATE', 'NEXT_MONTH_REVENUE']).columns
    
    # Initialize the XGBoost regressor with the selected hyperparameters.
    xgb_model = XGBRegressor(
        input_cols=feature_columns,
        label_cols=['NEXT_MONTH_REVENUE'],
        output_cols=['NEXT_MONTH_REVENUE_PREDICTION'],
        n_estimators=100,
        learning_rate=0.05,
        random_state=0
    )
    
    # Fit the model on the training data.
    xgb_model = xgb_model.fit(train_df)
    
    # ------------------------------------------------------------------------------
    # Evaluate Model Performance
    # ------------------------------------------------------------------------------
    print('Evaluating model...')
    
    # Generate predictions on the test set.
    predictions = xgb_model.predict(test_df)
    
    # Calculate the Mean Absolute Percentage Error (MAPE) as a performance metric.
    mape = mean_absolute_percentage_error(
        df=predictions, 
        y_true_col_names="NEXT_MONTH_REVENUE", 
        y_pred_col_names="NEXT_MONTH_REVENUE_PREDICTION"
    )
    
    # ------------------------------------------------------------------------------
    # Register the New Model Version
    # ------------------------------------------------------------------------------
    # Initialize the Registry instance for model registration.
    reg = Registry(
        session=session, 
        database_name=session.get_current_database(), 
        schema_name='MODEL_REGISTRY', 
        options={'enable_monitoring': True},
    )
    
    # Log (register) the model along with metrics, comments, and metadata.
    registered_model = reg.log_model(
        xgb_model,
        model_name="CUSTOMER_REVENUE_MODEL",
        version_name=model_version,
        metrics={
            'MAPE': mape, 
            'FEATURE_IMPORTANCE': dict(
                zip(
                    feature_columns, 
                    xgb_model.to_xgboost().feature_importances_.astype('float')
                )
            ),
            'TRAINING_DATA': {'FEATURE_CUTOFF_DATE': feature_cutoff_date}
        },
        comment="Model trained using XGBoost to predict revenue per customer for next month.",
        conda_dependencies=['xgboost'],
        sample_input_data=train_df.select(feature_columns).limit(10),
        options={"relax_version": False, "enable_explainability": True}
    )
    print(f'Registered new model with version {model_version} in model registry.')
    
    # ------------------------------------------------------------------------------
    # Evaluate Against the Production Model
    # ------------------------------------------------------------------------------
    # Compare the new model against the current production model.
    # This function may promote the new model if it outperforms the current one.
    evaluate_against_production_model(session, model_version, test_df)
    
    # ------------------------------------------------------------------------------
    # Save Baseline Predictions for Model Monitoring
    # ------------------------------------------------------------------------------
    # Cast columns to the appropriate data types.
    predictions = predictions.with_column('FEATURE_CUTOFF_DATE', F.col('FEATURE_CUTOFF_DATE').cast('timestamp'))
    predictions = predictions.with_column('NEXT_MONTH_REVENUE_PREDICTION', F.col('NEXT_MONTH_REVENUE_PREDICTION').cast('number(38,2)'))
    predictions = predictions.with_column('NEXT_MONTH_REVENUE', F.col('NEXT_MONTH_REVENUE').cast('number(38,2)'))
    
    # Save the baseline predictions to a Snowflake table.
    predictions.write.save_as_table(
        f'{session.get_current_database()}.MODEL_REGISTRY.MM_REVENUE_BASELINE_{model_version}', 
        mode='overwrite'
    )
    print(f'Baseline table for model monitor created with predictions until {feature_cutoff_date}.')
    
    # ------------------------------------------------------------------------------
    # Create Source Predictions for Model Monitoring
    # ------------------------------------------------------------------------------
    # Retrieve the feature dataframe for the prediction period using the target_end_date.
    feature_df = get_feature_df(session, feature_cutoff_date=target_end_date)
    
    # Run predictions using the registered model.
    predictions = registered_model.run(feature_df, function_name='PREDICT')
    
    # Cast columns to the appropriate data types.
    predictions = predictions.with_column('FEATURE_CUTOFF_DATE', F.col('FEATURE_CUTOFF_DATE').cast('timestamp'))
    predictions = predictions.with_column('NEXT_MONTH_REVENUE_PREDICTION', F.col('NEXT_MONTH_REVENUE_PREDICTION').cast('number(38,2)'))
    
    # Write the source predictions to a table for model monitoring.
    predictions.write.save_as_table(
        table_name=f'{session.get_current_database()}.MODEL_REGISTRY.MM_TRANS_SOURCE_{model_version}', 
        mode='overwrite'
    )
    print(f'Source table for model monitor created with predictions between {target_start_date} and {target_end_date}.')
    
    # ------------------------------------------------------------------------------
    # Create a Model Monitor to Track Model Performance Over Time
    # ------------------------------------------------------------------------------
    mm_db = session.get_current_database()
    session.sql(f"""
    CREATE OR REPLACE MODEL MONITOR {mm_db}.MODEL_REGISTRY.MM_{model_version} WITH
        MODEL={mm_db}.MODEL_REGISTRY.CUSTOMER_REVENUE_MODEL VERSION={model_version} FUNCTION=PREDICT
        SOURCE={mm_db}.MODEL_REGISTRY.MM_TRANS_SOURCE_{model_version}
        BASELINE={mm_db}.MODEL_REGISTRY.MM_REVENUE_BASELINE_{model_version},
        TIMESTAMP_COLUMN='FEATURE_CUTOFF_DATE'
        ID_COLUMNS=('CUSTOMER_ID')
        PREDICTION_SCORE_COLUMNS=('NEXT_MONTH_REVENUE_PREDICTION')
        ACTUAL_SCORE_COLUMNS=('NEXT_MONTH_REVENUE')
        WAREHOUSE=COMPUTE_WH
        REFRESH_INTERVAL='1 minute'
        AGGREGATION_WINDOW='1 day'
    """).collect()
    print(f'Model monitor created.')
    # Enable once 1.7.3 with bugfix is available
    # source_config = ModelMonitorSourceConfig(
    #     source=f'MLOPS_DEMO.MODEL_REGISTRY.MM_TRANS_SOURCE_{model_version}',
    #     timestamp_column='FEATURE_CUTOFF_DATE',
    #     id_columns=['CUSTOMER_ID'],
    #     prediction_score_columns=['NEXT_MONTH_REVENUE_PREDICTION'],
    #     actual_score_columns=['NEXT_MONTH_REVENUE'],
    #     baseline=f'MLOPS_DEMO.MODEL_REGISTRY.MM_REVENUE_BASELINE_{model_version}'
    # )
    
    # monitor_config = ModelMonitorConfig(
    #     model_version=reg.get_model('CUSTOMER_REVENUE_MODEL').version('PRODUCTION'),
    #     model_function_name='predict',
    #     background_compute_warehouse_name='COMPUTE_WH',
    #     refresh_interval='1 minute',
    #     aggregation_window='1 day'
    # )
    
    # reg.add_monitor(
    #     name=f'MLOPS_DEMO.MODEL_REGISTRY.MM_{model_version}',
    #     source_config=source_config,
    #     model_monitor_config=monitor_config
    # )
    return
