import streamlit as st
import ast
import pandas as pd
from snowflake.snowpark.functions import lit
import plotly.graph_objects as go

class ModelRegistryHelper:
    def __init__(self, session, registry):
        self.session = session
        self.registry = registry

        # Allowed model performance metrics
        self.ALLOWED_MODEL_TYPES_METRICS = {
            'TASK.TABULAR_BINARY_CLASSIFICATION': ['PRECISION', 'F1_SCORE', 'CLASSIFICATION_ACCURACY', 'ROC_AUC', 'RECALL'],
            'Task.TABULAR_REGRESSION': ['MSE', 'RMSE', 'MAPE', 'MAE']
        }
        
        # Allowed drift metrics
        self.ALLOWED_DRIFT_METRICS = ['JENSEN_SHANNON', 'WASSERSTEIN', 'DIFFERENCE_OF_MEANS']

    # Function to retrieve performance metrics for models from the model registry
    def get_model_performance_metrics(self, models, metrics, start_date, end_date, aggregation):
        """
        Fetches model performance metrics from the model registry for given models and metrics.
        
        Args:
            session: Active database session.
            models (list): List of models to fetch metrics for.
            metrics (list): List of performance metrics to retrieve.
            start_date (str): Start date for metrics retrieval.
            end_date (str): End date for metrics retrieval.
            aggregation (str): Aggregation window (e.g., '1 day').
        
        Returns:
            DataFrame: Aggregated model performance metrics.
        """
        
        all_models_metrics = []
        
        # Iterate through each model
        for model in models:
            model_name = model.model_name
            model_version_name = model.version_name
            monitor_name = str(self.registry.get_monitor(model_version=model).name)
            
            model_metrics_dfs = []
            
            # Fetch each metric for the model
            for metric in metrics:
                df_metric = (
                    self.session.table_function(
                        "MODEL_MONITOR_PERFORMANCE_METRIC",
                        lit(monitor_name), lit(metric), lit(aggregation), lit(start_date), lit(end_date)
                    )
                    .with_column('MODEL_NAME', lit(model_name))
                    .with_column('MODEL_VERSION_NAME', lit(model_version_name))
                    .with_column('MODEL_MONITOR_NAME', lit(monitor_name))
                    .rename({'METRIC_VALUE': metric})
                    .select(['MODEL_NAME', 'MODEL_VERSION_NAME', 'MODEL_MONITOR_NAME', 'EVENT_TIMESTAMP', metric])
                )
                model_metrics_dfs.append(df_metric)
            
            # Combine metrics for the model
            model_metrics_df = model_metrics_dfs[0]
            for df in model_metrics_dfs[1:]:
                model_metrics_df = model_metrics_df.join(df, on=['MODEL_NAME', 'MODEL_VERSION_NAME', 'MODEL_MONITOR_NAME', 'EVENT_TIMESTAMP'], how='inner')
            
            all_models_metrics.append(model_metrics_df)
        
        # Combine all models' metrics
        final_df = all_models_metrics[0]
        for df in all_models_metrics[1:]:
            final_df = final_df.union_all(df)
        
        return final_df.order_by(['MODEL_NAME', 'MODEL_VERSION_NAME', 'MODEL_MONITOR_NAME', 'EVENT_TIMESTAMP'])
    
    # Function to retrieve drift metrics for features from the model registry
    def get_model_drift_metrics(self, models, metrics, start_date, end_date, aggregation, columns):
        """
        Fetches drift metrics from the model registry for given models and metrics.
        
        Args:
            session: Active database session.
            models (list): List of models to fetch metrics for.
            metrics (list): List of performance metrics to retrieve.
            start_date (str): Start date for metrics retrieval.
            end_date (str): End date for metrics retrieval.
            aggregation (str): Aggregation window (e.g., '1 day').
            columns (list): List of columns to fetch metrics for.
        
        Returns:
            DataFrame: Aggregated drift metrics.
        """
        
        # Validate requested metrics
        invalid_metrics = set(metrics) - set(self.ALLOWED_DRIFT_METRICS)
        if invalid_metrics:
            raise ValueError(f"Invalid metric(s) found: {invalid_metrics}")
        
        all_models_metrics = []
        
        # Iterate through each model
        for model in models:
            model_name = model.model_name
            monitor_name = str(self.registry.get_monitor(model_version=model).name)
            
            model_metrics_dfs = []
            
            # Fetch each metric for the model
            for column in columns:
                column_metrics_dfs = []
                for metric in metrics:
                    df_metric = (
                        self.session.table_function(
                            "MODEL_MONITOR_DRIFT_METRIC",
                            lit(monitor_name), lit(metric), lit(column), lit(aggregation), lit(start_date), lit(end_date)
                        )
                        .with_column('MODEL_NAME', lit(model_name))
                        .with_column('MODEL_VERSION_NAME', lit(model.version_name))
                        .with_column('MODEL_MONITOR_NAME', lit(monitor_name))
                        .rename({'METRIC_VALUE': metric})
                        .select(['MODEL_NAME', 'MODEL_VERSION_NAME', 'MODEL_MONITOR_NAME', 'EVENT_TIMESTAMP', 'COLUMN_NAME', metric])
                    )
                    column_metrics_dfs.append(df_metric)
                column_metrics_df = column_metrics_dfs[0]
                for df in column_metrics_dfs[1:]:
                    column_metrics_df = column_metrics_df.join(df, on=['MODEL_NAME', 'MODEL_VERSION_NAME', 'MODEL_MONITOR_NAME', 'EVENT_TIMESTAMP', 'COLUMN_NAME'], how='inner')
                model_metrics_dfs.append(column_metrics_df)

            # Combine metrics for the model
            model_metrics_df = model_metrics_dfs[0]
            for df in model_metrics_dfs[1:]:
                model_metrics_df = model_metrics_df.union_all(df)
        
            all_models_metrics.append(model_metrics_df)
        
        # Combine all models' metrics
        final_df = all_models_metrics[0]
        for df in all_models_metrics[1:]:
            final_df = final_df.union_all(df)
        
        return final_df.order_by(['MODEL_NAME', 'MODEL_VERSION_NAME', 'MODEL_MONITOR_NAME', 'COLUMN_NAME', 'EVENT_TIMESTAMP'])

    def get_all_models(self):
        all_models = self.registry.show_models()
        all_models['model_task'] = all_models['name'].apply(lambda x: str(self.registry.get_model(x).version('default').get_model_task()))
        all_models['versions'] = all_models['versions'].apply(lambda x: ast.literal_eval(x))
        all_models['aliases'] = all_models['aliases'].apply(lambda x: ast.literal_eval(x))
        all_models = all_models.explode('versions')
        all_models = all_models.rename(columns={'versions': 'model_version', 'name': 'model_name'})
        all_models = all_models.sort_values(['model_name', 'created_on', 'model_version'])
        all_models = all_models[['model_name', 'model_version', 'aliases', 'model_task']]
        self.all_models = all_models
        return all_models

    def get_all_monitors(self):
        all_monitors = pd.DataFrame(self.registry.show_model_monitors())
        all_monitors['model'] = all_monitors['model'].apply(lambda x: ast.literal_eval(x))
        all_monitors['source'] = all_monitors['source'].apply(lambda x: ast.literal_eval(x))
        all_monitors['model_name'] = all_monitors['model'].apply(lambda x: x['model_name'])
        all_monitors['model_version'] = all_monitors['model'].apply(lambda x: x['version_name'])
        all_monitors['monitor_columns'] = all_monitors['source'].apply(lambda x: self.session.table(f"{x['database_name']}.{x['schema_name']}.{x['name']}").columns)
        all_monitors = all_monitors[['model_name', 'model_version', 'monitor_columns']]
        
        self.all_monitors = all_monitors
        return all_monitors

    def update_all_models(self):
        self.get_all_models()

    def update_all_monitors(self):
        self.get_all_monitors()

    def update_registry_data(self):
        self.get_all_models()
        self.get_all_monitors()

    def plot_model_performance(self):
        if not hasattr(self, 'all_models'):
            self.get_all_models()
        if not hasattr(self, 'all_monitors'):
            self.get_all_monitors()

        all_models = self.all_models
        all_monitors = self.all_monitors
            
        with st.expander('Select Models:', expanded=True):
            selection = st.dataframe(all_models, selection_mode='multi-row', on_select="rerun", hide_index=True, use_container_width=True)
        
        if len(selection['selection']['rows']) == 0:
            st.info('Select models.')
        else:
            selected_models = all_models.iloc[selection['selection']['rows']]
            if selected_models['model_task'].nunique() > 1:
                st.error('All selected models must have the same task.')
            else:
                with st.form("my_form"):
                    col1, col2 = st.columns(2)
                    if selected_models.iloc[0]['model_task'] == 'Task.TABULAR_REGRESSION':
                        selected_performance_metric = col1.selectbox('Select Model Performance Metric:', self.ALLOWED_MODEL_TYPES_METRICS['Task.TABULAR_REGRESSION'])
                    else:
                        selected_performance_metric = col1.selectbox('Select Model Performance Metric:', self.ALLOWED_MODEL_TYPES_METRICS['Task.TABULAR_BINARY_CLASSIFICATION'])
                    selected_drift_metric = col2.selectbox('Select Model Drift Metric:', self.ALLOWED_DRIFT_METRICS)

                    models = selected_models.apply(lambda row: self.registry.get_model(row['model_name']).version(row['model_version']), axis=1).tolist()
                    
                    selected_columns = st.multiselect('Select Drift columns:', all_monitors['monitor_columns'].explode().unique())
                    submitted = st.form_submit_button("Visualize Model Performance")
                    
                    if submitted:
                        df_model = self.get_model_performance_metrics(
                            models=models,
                            metrics=[selected_performance_metric],
                            start_date='2024-01-01',
                            end_date='2024-12-31',
                            aggregation='1 day'
                        ).to_pandas()
        
                        df_drift = self.get_model_drift_metrics(
                            models=models,
                            metrics=[selected_drift_metric],
                            start_date='2024-01-01',
                            end_date='2024-12-31',
                            aggregation='1 day',
                            columns=selected_columns
                        ).to_pandas()

                        df_drift["EVENT_TIMESTAMP"] = pd.to_datetime(df_drift["EVENT_TIMESTAMP"])
                        df_model["EVENT_TIMESTAMP"] = pd.to_datetime(df_model["EVENT_TIMESTAMP"])
        
                        fig = go.Figure()
                        
                        for model_version in df_model["MODEL_VERSION_NAME"].unique():
                            df_subset = df_model[df_model["MODEL_VERSION_NAME"] == model_version]
                            fig.add_trace(go.Scatter(
                                x=df_subset["EVENT_TIMESTAMP"],
                                y=df_subset[selected_performance_metric],
                                mode='lines+markers',
                                line=dict(dash='solid', width=4),
                                marker=dict(symbol='diamond', size=12),
                                name=f"{df_subset.iloc[0]['MODEL_NAME']} - {model_version}",
                                yaxis='y1',
                                legendgroup='model_metrics',
                                legendgrouptitle_text='Model Metrics:'
                            ))
                        
                        for model_version in df_drift["MODEL_VERSION_NAME"].unique():
                            df_subset = df_drift[df_drift["MODEL_VERSION_NAME"] == model_version]
                            for column_name in df_subset["COLUMN_NAME"].unique():
                                df_subsubset = df_subset[df_subset["COLUMN_NAME"] == column_name]
                                fig.add_trace(go.Scatter(
                                    x=df_subsubset["EVENT_TIMESTAMP"],
                                    y=df_subsubset[selected_drift_metric],
                                    mode='lines+markers',
                                    line=dict(dash='dot', width=2),
                                    marker=dict(symbol='square', size=8),
                                    name=f'{column_name}',
                                    yaxis='y2',
                                    legendgroup=f"{df_subsubset.iloc[0]['MODEL_NAME']} - {df_subsubset.iloc[0]['MODEL_VERSION_NAME']}",
                                    legendgrouptitle_text=f"Drift: {df_subsubset.iloc[0]['MODEL_NAME']} - {df_subsubset.iloc[0]['MODEL_VERSION_NAME']}"
                                ))
        
                        fig.update_layout(
                            title = {'text': "Model Performance & Feature Drift Over Time", 'font': {'size':24,'family':'Arial, sans-serif'}, 'xanchor':'center', 'x':0.5},
                            xaxis_title="Event Timestamp",
                            xaxis=dict(type='date'),
                            yaxis=dict(title=selected_performance_metric, side="left", showgrid=False),
                            yaxis2=dict(title=selected_drift_metric, overlaying="y", side="right", showgrid=False),
                            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, traceorder="grouped", itemwidth=30),
                            margin=dict(t=50),
                            legend_tracegroupgap=10,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig)