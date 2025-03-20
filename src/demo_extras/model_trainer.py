from snowflake.snowpark.functions import lit, col

# Snowflake Snowpark imports
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import (
    FeatureStore,
    CreationMode
)
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.metrics import mean_absolute_percentage_error
from snowflake.ml.monitoring.entities.model_monitor_config import ModelMonitorSourceConfig, ModelMonitorConfig
from opentelemetry import trace

import logging
from snowflake import telemetry
from opentelemetry import trace
import uuid

class ModelTrainer():
    def __init__(self, session):
        self.session = session
        self.registry = Registry(
            session=session, 
            database_name='SIMPLE_MLOPS_DEMO',
            schema_name='MODEL_REGISTRY', 
            options={'enable_monitoring': True},
        )
        self.fs = FeatureStore(
            session=session, 
            database='SIMPLE_MLOPS_DEMO', 
            name='FEATURE_STORE', 
            default_warehouse=session.get_current_warehouse(),
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self.logger = logging.getLogger("logger.ModelTrainer")
        self.tracer = trace.get_tracer("tracer.ModelTrainer")

        # Setting log levels for Snowflake loggers
        for logger_name in ['snowflake.connector']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)

    def train_new_model(self, feature_views: dict, feature_cutoff_date: str, target_start_date: str, target_end_date: str, model_version: str):
        if model_version == '':
            model_version = f"RAND_{str(uuid.uuid4()).upper().replace('-','_')}"
        train_df, test_df, feature_columns = self.prepare_data(feature_views, feature_cutoff_date, target_start_date, target_end_date, model_version)
        model = self.train(train_df, feature_columns)
        mape, predictions = self.evaluate_model(model, test_df)
        registered_model = self.register_new_model(model, model_version, train_df, feature_columns, feature_cutoff_date, mape)
        self.create_model_monitor(registered_model, model_version, predictions)
        self.evaluate_against_production_model(registered_model, test_df, mape)
        return mape

    def prepare_data(self, feature_views: dict, feature_cutoff_date: str, target_start_date: str, target_end_date: str, model_version: str):
        with self.tracer.start_as_current_span("Data Preparation"):
            feature_views = [self.fs.get_feature_view(fv,feature_views[fv]) for fv in feature_views]
            target_df = self.session.table('SIMPLE_MLOPS_DEMO.RETAIL_DATA.TRANSACTIONS')
            target_df = (
                target_df.filter(col('DATE').between(target_start_date,target_end_date))
                .group_by('CUSTOMER_ID')
                .agg(F.sum('TRANSACTION_AMOUNT').as_('NEXT_MONTH_REVENUE'))
                .with_column('FEATURE_CUTOFF_DATE', F.to_date(lit(feature_cutoff_date)))
            )
            
            customers_df = self.session.table('SIMPLE_MLOPS_DEMO.RETAIL_DATA.CUSTOMERS').select('CUSTOMER_ID').distinct()
            spine_df = target_df.join(customers_df, on=['CUSTOMER_ID'], how='outer')
            spine_df = spine_df.fillna(0, subset='NEXT_MONTH_REVENUE')
    
            train_dataset = self.fs.generate_dataset(
                name="SIMPLE_MLOPS_DEMO.FEATURE_STORE.NEXT_MONTH_REVENUE_DATASET",
                spine_df=spine_df,
                features=feature_views,
                version=model_version,
                spine_timestamp_col="FEATURE_CUTOFF_DATE",
                spine_label_cols=["NEXT_MONTH_REVENUE"],
                include_feature_view_timestamp_col=False,
                desc=f"Training dataset from {feature_cutoff_date}"
            )
            
            df = train_dataset.read.to_snowpark_dataframe()
            train_df, test_df = df.random_split(weights=[0.9, 0.1], seed=0)
            feature_columns = train_df.drop(['CUSTOMER_ID', 'FEATURE_CUTOFF_DATE', 'NEXT_MONTH_REVENUE']).columns
            self.logger.info('Training dataset created.')
            return  train_df, test_df, feature_columns
        
    def train(self, train_df, feature_columns):
        with self.tracer.start_as_current_span("Model Fitting"):
            model = XGBRegressor(
                input_cols=feature_columns,
                label_cols=['NEXT_MONTH_REVENUE'],
                output_cols=['NEXT_MONTH_REVENUE_PREDICTION'],
                n_estimators=100,
                learning_rate=0.05,
                random_state=0
            )
            model = model.fit(train_df)
            self.logger.info('Successfully trained a new model.')
            return model

    def evaluate_model(self, model, test_df):
        with self.tracer.start_as_current_span("Model Evaluation"):
            predictions = model.predict(test_df)
            mape = mean_absolute_percentage_error(
                df=predictions, 
                y_true_col_names="NEXT_MONTH_REVENUE", 
                y_pred_col_names="NEXT_MONTH_REVENUE_PREDICTION"
            )
            telemetry.add_event("model_evaluation.done", {"metric": "mape", "value": mape})
            self.logger.info(f'New model has a MAPE of {mape}.')
            return mape, predictions

    def register_new_model(self, model, model_version, train_df, feature_columns, feature_cutoff_date, mape):
        with self.tracer.start_as_current_span("Model Registration"):
            registered_model = self.registry.log_model(
                model,
                model_name="CUSTOMER_REVENUE_MODEL",
                version_name=model_version,
                metrics={
                    'MAPE': mape, 
                    'TRAINING_DATA': {'FEATURE_CUTOFF_DATE': feature_cutoff_date}
                },
                comment="Model trained using XGBoost to predict revenue per customer for next month.",
                conda_dependencies=['xgboost'],
                sample_input_data=train_df.select(feature_columns).limit(10),
                options={"relax_version": False, "enable_explainability": True}
            )
            self.logger.info(f'Registered new model with version {model_version} in model registry.')
            return registered_model

    def create_model_monitor(self, registered_model, model_version, predictions):
        with self.tracer.start_as_current_span("Model Monitor Creation"):
            predictions = predictions.with_column('FEATURE_CUTOFF_DATE', F.col('FEATURE_CUTOFF_DATE').cast('timestamp'))
            predictions.write.save_as_table(f'SIMPLE_MLOPS_DEMO.MODEL_REGISTRY.MM_REVENUE_BASELINE_{model_version}', mode='overwrite')
            predictions.write.save_as_table(f'SIMPLE_MLOPS_DEMO.MODEL_REGISTRY.MM_REVENUE_SOURCE_{model_version}', mode='overwrite')
            
            source_config = ModelMonitorSourceConfig(
                source=f'SIMPLE_MLOPS_DEMO.MODEL_REGISTRY.MM_REVENUE_SOURCE_{model_version}',
                baseline=f'SIMPLE_MLOPS_DEMO.MODEL_REGISTRY.MM_REVENUE_BASELINE_{model_version}',
                timestamp_column='FEATURE_CUTOFF_DATE',
                id_columns=['CUSTOMER_ID'],
                prediction_score_columns=['NEXT_MONTH_REVENUE_PREDICTION'],
                actual_score_columns=['NEXT_MONTH_REVENUE'],
            )
            
            monitor_config = ModelMonitorConfig(
                model_version=registered_model,
                model_function_name='predict',
                background_compute_warehouse_name='COMPUTE_WH',
                refresh_interval='1 minute',
                aggregation_window='1 day'
            )
            
            model_monitor = self.registry.add_monitor(
                name=f'SIMPLE_MLOPS_DEMO.MODEL_REGISTRY.MM_{model_version}',
                source_config=source_config,
                model_monitor_config=monitor_config
            )

    def evaluate_against_production_model(self, registered_model, test_df, mape):
        with self.tracer.start_as_current_span("Model Deployment"):
            production_model = self.registry.get_model('CUSTOMER_REVENUE_MODEL').version('PRODUCTION')
            production_model_predictions = production_model.run(test_df, function_name='PREDICT')
            production_model_mape = mean_absolute_percentage_error(
                df=production_model_predictions, 
                y_true_col_names="NEXT_MONTH_REVENUE", 
                y_pred_col_names="NEXT_MONTH_REVENUE_PREDICTION"
            )
            
            if mape < production_model_mape:
                self.logger.info(f"New model has a lower MAPE compared to current production model.")
                self.logger.info(f"New model will be put into production by setting its alias to PRODUCTION.")
                
                # Update model aliases:
                production_model.unset_alias('PRODUCTION')
                production_model.set_alias('DEPRECATED')
                registered_model.set_alias('PRODUCTION')
                telemetry.add_event("model_deployment.done", {"mape_old": production_model_mape, "mape_new": mape})
            else:
                telemetry.add_event("model_deployment.skipped", {"mape_old": production_model_mape, "mape_new": mape})
                self.logger.warn(f"Existing production model has a lower MAPE compared to the developed model.")
                self.logger.warn(f"New model is not automatically set into production.")