# Disable warnings from model registry when model from model registry is loaded into notebook
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Snowflake Snowpark imports
from snowflake.core import Root, CreateMode
from snowflake.core.schema import Schema
from snowflake.snowpark.functions import lit, col
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import (
    FeatureStore,
    CreationMode
)

# Third-party imports
import pandas as pd
import time
from datetime import datetime
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
from snowflake.core.stage import Stage, StageEncryption
import streamlit as st
import plotly.express as px

from snowflake.snowpark import functions as F

class Demoflow():
    def __init__(self):
        self.session = get_active_session()
        self.root = Root(self.session)
        self.fs = FeatureStore(
            session=self.session, 
            database='SIMPLE_MLOPS_DEMO', 
            name='FEATURE_STORE', 
            default_warehouse='FEATURE_STORE_WH',
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )

    def setup(self):
        # Setting up data for demo SNOWFLAKE_SSE
        self.root.databases['SIMPLE_MLOPS_DEMO'].schemas.create(schema=Schema(name="RETAIL_DATA"), mode=CreateMode.or_replace)
        self.root.databases['SIMPLE_MLOPS_DEMO'].schemas.create(schema=Schema(name="RETAIL_DATA"), mode=CreateMode.or_replace)
        self.root.databases['SIMPLE_MLOPS_DEMO'].schemas.create(schema=Schema(name="FEATURE_STORE"), mode=CreateMode.or_replace)
        self.root.databases['SIMPLE_MLOPS_DEMO'].schemas.create(schema=Schema(name="MODEL_REGISTRY"), mode=CreateMode.or_replace)
        self.root.databases["SIMPLE_MLOPS_DEMO"].schemas["PUBLIC"].stages.create(stage=Stage(name="PIPELINES", encryption=StageEncryption(type="SNOWFLAKE_FULL"), comment='Stage for storing pipelines.'), mode=CreateMode.or_replace)
        self.session.table('SIMPLE_MLOPS_DEMO._DATA_GENERATION._TRANSACTIONS').filter(col('DATE') <= lit('2024-04-30')).write.save_as_table(table_name='SIMPLE_MLOPS_DEMO.RETAIL_DATA.TRANSACTIONS', mode='overwrite')
        self.session.table('SIMPLE_MLOPS_DEMO._DATA_GENERATION._CUSTOMERS').write.save_as_table('SIMPLE_MLOPS_DEMO.RETAIL_DATA.CUSTOMERS')
        print('Setup finished.')

    def _generate_date_list(self, start_date, end_date):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        list_of_dates = []
        current_date = start
        
        while current_date <= end:
            list_of_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += relativedelta(months=1)
    
        return list_of_dates

    def _get_feature_df(self, model, feature_cutoff_date):
        # Use lineage information to retrieve the feature views of this model
        feature_views = model.lineage(direction='upstream')[0].lineage(domain_filter=['feature_view'], direction='upstream')
        
        # Create the spine dataframe containing all customers
        spine_df = (
            self.session.table('SIMPLE_MLOPS_DEMO.RETAIL_DATA.CUSTOMERS')
            .select('CUSTOMER_ID')
            .with_column('FEATURE_CUTOFF_DATE', F.to_date(lit(feature_cutoff_date)))
        )
    
        # Retrieve feature values from the Feature Store for the specified cutoff date.
        feature_df = self.fs.retrieve_feature_values(
            spine_df=spine_df,
            features=feature_views,
            spine_timestamp_col="FEATURE_CUTOFF_DATE"
        )
        return feature_df

    def _is_feature_store_updated(self, model, ts):
        feature_view_refresh_histories = []
        for feature_view in model.lineage(direction='upstream')[0].lineage(domain_filter=['feature_view'], direction='upstream'):
            features_refreshed = (
                self.fs.get_refresh_history(feature_view)
                .order_by(col('REFRESH_END_TIME').desc())
                .limit(1)
            )
            feature_view_refresh_histories.append(features_refreshed)
        
        # join all histories
        feature_view_refresh_histories = reduce(lambda df1, df2: df1.union_all(df2), feature_view_refresh_histories)
        feature_view_refresh_histories = feature_view_refresh_histories.cross_join(ts)
        feature_view_refresh_histories = feature_view_refresh_histories.with_column('UPDATED', col('REFRESH_END_TIME') > col('TIMESTAMP'))
        if feature_view_refresh_histories.filter(col('UPDATED') == False).count() > 0:
            return False
        else:
            return True

    def _wait_until_feature_store_updated(self, model, interval=3):
        """
        Waits until the Feature Store has been updated relative to the current timestamp.
        """
        # Create a one-row dataframe containing the current timestamp.
        ts = self.session.range(1).with_column('TIMESTAMP', F.current_timestamp()).drop('ID').cache_result()

        start_time = time.time()
        # Continuously check if the feature store has been updated.
        while not self._is_feature_store_updated(model, ts):
            # Print a dynamic status message with elapsed time (overwriting the same line).
            print(f"\rWaiting for updated Feature Store ... ({int(time.time() - start_time)} seconds.)", end="", flush=True)
            # Pause for the specified interval before checking again.
            time.sleep(interval)

    def simulate_model_performance(self, model, start_date, end_date, generate_data=False):
        if generate_data:
            # Add future transactions
            new_transactions = self.session.table(f'SIMPLE_MLOPS_DEMO._DATA_GENERATION._TRANSACTIONS') \
                                    .filter(col('DATE').between(start_date, end_date))
            new_transactions.write.save_as_table(
                table_name=f'SIMPLE_MLOPS_DEMO.RETAIL_DATA.TRANSACTIONS', 
                mode='append'
            )

        # Retrieve the specific model version from the registry.
        registry = Registry(
            session=self.session, 
            database_name='SIMPLE_MLOPS_DEMO', 
            schema_name='MODEL_REGISTRY', 
            options={'enable_monitoring': True},
        )

        # Retrieve the source table of the model version's model monitor
        model_monitors = pd.DataFrame(registry.show_model_monitors())
        model_monitor = model_monitors[model_monitors['name'] == registry.get_monitor(model_version=model).name]
        model_monitor_source = json.loads(model_monitor['source'].iloc[0])
        model_monitor_source = f"{model_monitor_source['database_name']}.{model_monitor_source['schema_name']}.{model_monitor_source['name']}"
        # Check if feature store is updated
        if generate_data:
            self._wait_until_feature_store_updated(model)
            print('')

        # Generate predictions and actuals per month
        for date in self._generate_date_list(start_date, end_date):
            # predictions
            feature_df = self._get_feature_df(
                model, 
                feature_cutoff_date=date, 
            )
            predictions = model.run(feature_df, function_name='PREDICT')
            predictions.write.save_as_table(table_name=model_monitor_source.split('.'), mode='append', column_order='name')
            print(f'Generated predictions with features until: {date}.')

            actual_values_df = (
                self.session.table('SIMPLE_MLOPS_DEMO.RETAIL_DATA.TRANSACTIONS')
                .filter(col('DATE').between(F.date_add(lit(date), 1), F.add_months(lit(date), lit(1))))
                .group_by(['CUSTOMER_ID'])
                .agg(F.sum('TRANSACTION_AMOUNT').as_('TOTAL_REVENUE'))
                .with_column('DATE',  F.to_date(lit(date)))
            )
            # Get list of all customers
            customers_df = self.session.table('SIMPLE_MLOPS_DEMO.RETAIL_DATA.CUSTOMERS').select('CUSTOMER_ID').distinct()
            
            # Assume 0 revenue for customers without transactions
            actual_values_df = actual_values_df.join(customers_df, on=['CUSTOMER_ID'], how='outer')
            actual_values_df = actual_values_df.fillna(0,subset='TOTAL_REVENUE')
            # Update source table from model monitor
            source_table = self.session.table(model_monitor_source)
            source_table.update(
                condition=(
                    (source_table['FEATURE_CUTOFF_DATE'] == actual_values_df['DATE']) &
                    (source_table['CUSTOMER_ID'] == actual_values_df['CUSTOMER_ID'])
                ),
                assignments={
                    "NEXT_MONTH_REVENUE": actual_values_df['TOTAL_REVENUE'],
                },
                source=actual_values_df
            )
            print(f"Generated actual values until: {(datetime.strptime(date, '%Y-%m-%d') + relativedelta(months=1)).strftime('%Y-%m-%d')}.")

    def get_customer_revenue_plot(self, snowpark_df, customer_id):
        df_filtered = snowpark_df.filter(col("CUSTOMER_ID") == customer_id)
        df_with_week = df_filtered.with_column("WEEK_START", F.date_trunc("week", col("DATE")))
        
        df_grouped = df_with_week.group_by("WEEK_START", "TRANSACTION_CHANNEL") \
            .agg(F.avg(col("TRANSACTION_AMOUNT")).alias("TRANSACTION_AMOUNT"))
        
        df_total = df_grouped.group_by("WEEK_START") \
            .agg(F.sum(col("TRANSACTION_AMOUNT")).alias("TOTAL_REVENUE"))
        
        df_joined = df_grouped.join(df_total, on="WEEK_START")
        df_final = df_joined.with_column("PERCENTAGE", 
                                        col("TRANSACTION_AMOUNT") / col("TOTAL_REVENUE") * 100)
        
        pdf = df_final.to_pandas()
        
        # Plot using Plotly Express
        fig = px.bar(
            pdf, 
            x="WEEK_START", 
            y="PERCENTAGE", 
            color="TRANSACTION_CHANNEL",
            title="Percentage Distribution of Average Transaction Amount per Calendar Week"
        )
        fig.update_xaxes(title="Week Start Date")
        fig.update_yaxes(title="Percentage")
        st.plotly_chart(fig)