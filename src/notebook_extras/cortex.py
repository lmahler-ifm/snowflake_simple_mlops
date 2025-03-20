import streamlit as st
import json
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete, CompleteOptions
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import approx_count_distinct, col
from snowflake.snowpark.window import Window
import inspect
import re
import pandas as pd

st.session_state['hist'] = []

SYSTEM_PROMPT_TEMPLATE_SUGGEST_PROMPTS = """
You will later be tasked to create Plotly charts and display them in Streamlit. 
But first the user wants to understand what kind of questions they can ask.
For that, you are provided the first 10 rows of a dataframe.

Make sure to suggest prompts that:
* Generate analytical insights based on dataframe transformations.
* Include instructions about how the plot should look.

Return the suggested prompts as JSON using the following format:
{prompts:[{"prompt": prompt, "prompt_explanation": prompt_explanation}]}
Only return the JSON, no other content.
"""

SYSTEM_PROMPT_TEMPLATE_CREATE_PLOT = """
You will be tasked to create Plotly charts and display them in Streamlit. 
The environment is already set up, so only return code to manipulate the given dataframe and afterwards plot it using Plotly and Streamlit.
The dataframe is of type {dataframe_type}.

If the dataframe is pandas, use pandas transformations.
If it is Snowpark, use Snowpark Python transformations.
When using Snowpark functions, import them using:
from snowflake.snowpark import functions as F.
Reference functions like:
F.sum(), F.max(), etc.

Before creating plots, make sure to order the relevant date/timestamp column if the dataframe has such a column.
When using df.to_pandas() on date columns, make sure to convert the date columns to pandas datetime like this:
df = df.to_pandas()
df['MONTH'] = pd.to_datetime(df['MONTH'])

When you create new dataframes, make sure that the object starts with an underscore.
Use df in your code to reference the dataframe. The dataframe has the following columns: {dataframe_columns}

The first 5 rows of the dataframe look like this:
{dataframe_sample}
"""

USER_PROMPT_TEMPLATE_DESCRIBE_COLUMN = """
Look at the provided sample of my dataframe and provide a short business description for each column in my dataframe.
Return the descriptions in the following JSON format:
{{"column_name1": "column_description1", "column_name2": "column_description2"}}

The first 5 rows of the dataframe look like this:
{dataframe_sample}

The available columns are:
{dataframe_columns}
"""

USER_PROMPT_TEMPLATE_DESCRIBE_COLUMN_SQL = """
Given the following SQL query, explain how the column {column} is calculated.
Make sure your explanation is easy to follow and also provide a short summary of the explanation.
The SQL Query:
{sql_query}
"""

USER_PROMPT_TEMPLATE_DESCRIBE_DATA = """
You are provided with a dataframe that has the following statistics for every column of my dataframe:
* count (total count of values)
* max (max value for numerical columns, first alphabetical value for categorical columns)
* mean (mean value for numerical columns, not applicable for categorical columns)
* min (min value for numerical columns, last alphabetical value for categorical columns)
* stddev (standard deviation for numerical columns, not applicable for categorical columns)
* top (most common value for categorical variables)
* freq (most common value’s frequency)
* unique (number of unique values for categorical columns, not applicable for numerical columns)
* datatype (string which is a categorical column, other values are numerical columns)

Based on this information, provide insights into potential data quality issues.

Specifically, please identify and discuss:  

1. **Missing Values:** Columns with significantly lower `count` values compared to others, which could indicate missing data.  
2. **Outliers:** Columns where `min` or `max` values are far from the `mean`, or where `stddev` is unusually high, suggesting extreme values.  
3. **Data Skewness:** Columns where the `mean` is significantly different from the `min` and `max`, indicating skewed distributions.  
4. **Potential Data Type Issues:** Columns where numerical statistics may indicate categorical or incorrectly formatted data.  
5. **Feature Scaling Issues:** Columns with very large or very small values that might require normalization or standardization for ML models.  
6. **Other Anomalies:** Any unusual patterns that could suggest data corruption, incorrect data entry, or inconsistencies.  

Provide actionable recommendations on how to address any detected issues to improve the dataset's quality.

Return these recommendations as a markdown table for streamlit's st.markdown() function with the following columns:
* Column Name
* Issue found (description of the actual issue found)
* Recommendation (description of steps recommended to mitigate the found issue)

Make sure there is one row per combination of column and issue.
Only return the markdown table.

Here is the output of `df.describe()`:  

{dataframe_sample}
"""

USER_PROMPT_TEMPLATE_DESCRIBE_DATA_FOR_ML = """
You are provided with a dataframe that has the following statistics for every column of my dataframe:
* count (total count of values)
* max (max value for numerical columns, first alphabetical value for categorical columns)
* mean (mean value for numerical columns, not applicable for categorical columns)
* min (min value for numerical columns, last alphabetical value for categorical columns)
* stddev (standard deviation for numerical columns, not applicable for categorical columns)
* top (most common value for categorical variables)
* freq (most common value’s frequency)
* unique (number of unique values for categorical columns, not applicable for numerical columns)
* datatype (string which is a categorical column, other values are numerical columns)

Based on this information, provide insights into potential data quality issues that could impact training of a machine learning model.  

Specifically, please identify and discuss:  

1. **Missing Values:** Columns with significantly lower `count` values compared to others, which could indicate missing data.  
2. **Outliers:** Columns where `min` or `max` values are far from the `mean`, or where `stddev` is unusually high, suggesting extreme values.  
3. **Data Skewness:** Columns where the `mean` is significantly different from the `min` and `max`, indicating skewed distributions.  
4. **Potential Data Type Issues:** Columns where numerical statistics may indicate categorical or incorrectly formatted data.  
5. **Feature Scaling Issues:** Columns with very large or very small values that might require normalization or standardization for ML models.  
6. **Other Anomalies:** Any unusual patterns that could suggest data corruption, incorrect data entry, or inconsistencies.  

Provide actionable recommendations on how to address any detected issues to improve the dataset's quality for machine learning.  
Make sure to base your recommendations on the type of model that will be trained which will be and the feature column:
Model: {model_type}
Target: {target_variable}

Return these recommendations as a markdown table for streamlit's st.markdown() function with the following columns:
* Column Name
* Issue found (description of the actual issue found)
* Recommendation (description of steps recommended to mitigate the found issue)

Make sure there is one row per combination of column and issue.
Only return the markdown table.

Here is the output of `df.describe()`:  

{dataframe_sample}
"""


class CortexPilot():
    def __init__(self, session=None, llm='mistral-large2', temperature=0, top_p=0):
        if session is None:
            self.session = get_active_session()
        else:
            self.session = session
        self.llm = llm
        self.llm_options = CompleteOptions(
            temperature=temperature,
            top_p=top_p
        )
        self.dataframe_type_icons = {
            "pandas.core.frame.DataFrame": "🐼",
            "snowflake.snowpark.dataframe.DataFrame": "❄️",
            "snowflake.snowpark.table.Table": "❄️"
        }
        st.session_state['suggested_prompts'] = None
        

    def _extract_python_code(self, text):
        """
        Function to extract Python code from LLM responses.
        """
        # Regular expression pattern to extract content between triple backticks with 'python' as language identifier
        pattern = r"```python(.*?)```"
        # re.DOTALL allows the dot (.) to match newlines as well
        match = re.search(pattern, text, re.DOTALL)
        if match is not None:
            return match.group(1).strip()
        else:
            return text.strip()

    def _extract_json_code(self, text):
        """
        Function to extract JSON contents from LLM responses.
        """
        pattern = r"```json(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match is not None:
            return match.group(1).strip()
        else:
            return text.strip()

    def _select_dataframe(self):
        """
        Identifies supported DataFrame variables available in the global scope.
        Allows the user to select one via Streamlit UI.
        """
        #
        global_notebook_variables = inspect.currentframe().f_back.f_back.f_globals
        # Retrieve DataFrame variables from the global scope
        available_dataframes = {
            var_name: (var_obj, self.dataframe_type_icons.get(f"{type(var_obj).__module__}.{type(var_obj).__name__}"))
            #for var_name, var_obj in globals().items()
            for var_name, var_obj in global_notebook_variables.items()
            if f"{type(var_obj).__module__}.{type(var_obj).__name__}" in self.dataframe_type_icons
        }
        
        # Create a dropdown for selecting a DataFrame
        dataframe_options = [f"[{icon}] {name}" for name, (_, icon) in available_dataframes.items() if not name.startswith("_")]
        selected_option = st.selectbox("Select DataFrame:", dataframe_options)
        
        # Extract the selected DataFrame's name
        selected_dataframe_name = selected_option.split("] ")[1]
        selected_dataframe, dataframe_type_icon = available_dataframes[selected_dataframe_name]
        
        # Convert the icon representation back to the actual DataFrame type
        dataframe_type = "snowflake.snowpark.dataframe.DataFrame" if dataframe_type_icon == "❄️" else "pandas.core.frame.DataFrame"
        
        # Display a sample of the selected DataFrame
        with st.expander("Sample Data:", expanded=True):
            try:
                self.sample_data = selected_dataframe.limit(10).to_pandas() if dataframe_type.startswith("snowflake") else selected_dataframe.head(10)
                st.dataframe(self.sample_data, use_container_width=True)
            except Exception as error:
                st.error("Error displaying sample data")
                st.error(error)
        
        return selected_dataframe

    def _suggest_llm_prompts(self):
        """
        Generates and displays suggested prompts for analyzing the selected DataFrame.
        """
        with st.form("Prompt Suggestions:", border=False):
            if st.form_submit_button("🤖 What can I ask?"):
                user_prompt = f"""
                I have the following data:
                {self.sample_data.to_markdown()}
                Suggest 3 prompts that I could use.
                """
                llm_input = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE_SUGGEST_PROMPTS}, {"role": "user", "content": user_prompt}]
                llm_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
                st.session_state['hist'].append(llm_response)
                suggested_prompts = json.loads(self._extract_json_code(llm_response))
                st.session_state['suggested_prompts'] = suggested_prompts
        if st.session_state['suggested_prompts'] is not None:
            with st.expander("Sample Questions:", expanded=True):
                for prompt in st.session_state['suggested_prompts']['prompts']:
                    with st.container(border=True):
                        st.code(prompt['prompt'], language=None)
                        st.markdown(f"**{prompt['prompt_explanation']}**")

    def _generate_plotly_code(self, df):
        """
        Asks the LLM to generate Plotly visualization code based on user input and the selected DataFrame.
        """
        with st.form("Ask LLM"):
            user_query = st.text_area("What can I help you with?", height=4*34)
            
            if st.form_submit_button("🤖 Ask Cortex!"):
                system_prompt = SYSTEM_PROMPT_TEMPLATE_CREATE_PLOT.format(dataframe_type=type(df).__module__, dataframe_columns=df.columns, dataframe_sample=self.sample_data.to_markdown())
                llm_input = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
                llm_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
                
                try:
                    generated_code = self._extract_python_code(llm_response)
                    exec(generated_code)
                    with st.expander("View Code generated by LLM"):
                        st.code(generated_code)
                except Exception as e:
                    st.info("First LLM response contained invalid code. Retrying with error context...")
                    with st.expander("View Error and LLM Response"):
                        st.error(e)
                        st.info(llm_response)
                    
                    llm_input.append({"role": "assistant", "content": llm_response})
                    llm_input.append({"role": "user", "content": f"The generated code resulted in an error: {str(e)}. Please adjust it."})
                    retry_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
                    
                    try:
                        retry_code = self._extract_python_code(retry_response)
                        exec(retry_code)
                        with st.expander("View Adjusted Code"):
                            st.code(retry_code)
                    except Exception as retry_error:
                        st.error("Adjusted code also contains errors.")
                        st.error(retry_error)

    def f_cortex_helper_visualize_query(self, df, user_query, verbose=False):
        """
        Asks the LLM to generate Plotly visualization code based on user input and the selected DataFrame.
        """
        if isinstance(df, pd.DataFrame):
            system_prompt = SYSTEM_PROMPT_TEMPLATE_CREATE_PLOT.format(dataframe_type=type(df).__module__, dataframe_columns=df.columns, dataframe_sample=df.head(5).to_markdown())
        else:
            system_prompt = SYSTEM_PROMPT_TEMPLATE_CREATE_PLOT.format(dataframe_type=type(df).__module__, dataframe_columns=df.columns, dataframe_sample=df.sample(n=5).to_pandas().to_markdown())
        
        llm_input = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        llm_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
        
        try:
            generated_code = self._extract_python_code(llm_response)
            exec(generated_code)
            with st.expander("View Code generated by LLM"):
                st.code(generated_code)
        except Exception as e:
            if verbose:
                st.info("First LLM response contained invalid code. Retrying with error context...")
                with st.expander("View Error and LLM Response"):
                    st.error(e)
                    st.info(llm_response)
            
            llm_input.append({"role": "assistant", "content": llm_response})
            llm_input.append({"role": "user", "content": f"The generated code resulted in an error: {str(e)}. Please adjust it."})
            retry_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
            
            try:
                retry_code = self._extract_python_code(retry_response)
                exec(retry_code)
                with st.expander("View Adjusted Code"):
                    st.code(retry_code)
            except Exception as retry_error:
                st.error("Adjusted code also contains errors.")
                st.error(retry_error)

    def ui_plotting(self):
        """
        Opens a UI to select a dataframe and use Cortex LLMs to automatically visualize data based on user prompts.
        """
        st.subheader('🤖 Ask Cortex about your Data! ', help='Select a dataframe and ask Cortex for generating plots.')
        dataframe = self._select_dataframe()
        self._suggest_llm_prompts()
        self._generate_plotly_code(dataframe)
    
    def f_describe_columns(self, df, columns: list = None, exclude_columns: list =None):
        """
        Function to use Cortex LLMs to generate business descriptions for columns of a dataframe.
        """
        if isinstance(df, pd.DataFrame):
            if columns:
                df = df[columns]
            if exclude_columns:
                df = df.drop(exclude_columns, axis=1)
        else:
            if columns:
                df = df.select(columns)
            if exclude_columns:
                df = df.drop(exclude_columns)
            df = df.sample(n=5).to_pandas()
        prompt = USER_PROMPT_TEMPLATE_DESCRIBE_COLUMN.format(dataframe_sample=df.head(5).to_markdown(), dataframe_columns=df.columns)
        llm_response = complete(self.llm, prompt, stream=False)
        llm_response = self._extract_json_code(llm_response)
        feature_descriptions = json.loads(llm_response)
        return feature_descriptions
    
    def f_explain_column_sql(self, column, sql_query):
        """
        Function to use Cortex LLMs to explain the calculation of a column based on the provided SQL.
        """
        #prompt = f'You are given a SQL query. Explain how the column {column} is calculated. The SQL query: {sql}'
        prompt = USER_PROMPT_TEMPLATE_DESCRIBE_COLUMN_SQL.format(column=column, sql_query=sql_query)
        resp = complete(self.llm, prompt)
        return resp

    def _analyze_unique_values(self, df):
        """
        Get unique counts per string column.
        """
        categorical_columns = [col[0] for col in df.dtypes if col[1].startswith("string")]
        df_unique_counts = df.select(
            [approx_count_distinct(col(c)).alias(f"{c}") for c in categorical_columns]
        ).to_pandas()
        
        df_unique_counts['SUMMARY'] = 'unique'
        return df_unique_counts
    
    def _analyze_column_datatypes(self, df, describe_columns):
        """
        Create a dataframe with column names and their datatype
        """
        df_dtypes = pd.DataFrame(df.select(describe_columns).dtypes)
        #df_dtypes[1] = df_dtypes[1].apply(lambda x: x.replace('(16777216)',''))
        df_dtypes[1] = df_dtypes[1].str.replace(r"\(.*\)", "", regex=True)
        df_dtypes = df_dtypes.set_index(0).T
        df_dtypes['SUMMARY'] = 'datatype'
        return df_dtypes
    
    def _describe(self, df):
        """
        Run describe on provided dataframe and return list of of columns
        """
        describe_result = df.describe().to_pandas().round(3)
        describe_columns = list(describe_result.drop('SUMMARY',axis=1).columns)
        return describe_result, describe_columns
    
    def _get_dataframe_description(self, df):
        """
        Describe dataframe using Snowpark describe() and enrich it with additional analysis.
        """
        describe_result, describe_columns = self._describe(df)
        df_unique_counts = self._analyze_unique_values(df)
        df_top_freq = self._get_freq_top_(df)
        df_dtypes = self._analyze_column_datatypes(df, describe_columns)
        df_describe = pd.concat([describe_result, df_top_freq, df_unique_counts, df_dtypes])
        df_describe = df_describe.where(pd.notna(df_describe), '')
        return df_describe

    def _get_freq_top_(self, df):
        categorical_columns = [col[0] for col in df.dtypes if col[1].startswith("string")]
        for col_i, _col in enumerate(categorical_columns):
            window_spec = Window.order_by(col("count").desc(), col(_col))
            _col_df = (
                df.group_by(_col)
                .agg(F.count("*").alias("count"))
                .with_column("rank", F.rank().over(window_spec))
                .filter(col("rank") == 1)
                .select(col(_col).alias('"top"'), col("count").alias('"freq"'))
            )
            if col_i == 0:
                top_freq_df = _col_df
            else:
                top_freq_df = top_freq_df.union_all_by_name(_col_df)
        top_freq_df = top_freq_df.to_pandas()
        top_freq_df.index = categorical_columns
        top_freq_df = top_freq_df.T
        top_freq_df = top_freq_df.reset_index(names='SUMMARY')
        return top_freq_df
    

    def f_cortex_helper_describe_data(self, df):
        """
        Requests the LLM to analyze potential issues in your data based on the output of Snowpark's `describe()` function.  
        Returns the analyzed dataframe along with a summary of detected issues.
        """
        if isinstance(df, pd.DataFrame):
            df = self.session.create_dataframe(df)
        df_describe = self._get_dataframe_description(df).reset_index(drop=True)
        #df_describe = df.describe().to_pandas()
        df_for_prompt = df_describe.to_markdown()
        user_query = USER_PROMPT_TEMPLATE_DESCRIBE_DATA.format(dataframe_sample = df_for_prompt)
        llm_input = [{"role": "user", "content": user_query}]
        llm_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
        return df_describe, llm_response
        
    def f_cortex_helper_describe_data_for_ml(self, df, target_column, model_type='XGBoost Classifier'):
        """
        Requests the LLM to analyze potential issues in your data when used as training data for a machine learning model, based on the output of Snowpark's `describe()` function.  
        Returns the analyzed dataframe along with a summary of detected issues and recommendations for improvement given a model type.
        """
        if isinstance(df, pd.DataFrame):
            df = self.session.create_dataframe(df)
        df_describe = self._get_dataframe_description(df).reset_index(drop=True)
        #df_describe = df.describe().to_pandas()
        df_for_prompt = df_describe.to_markdown()
        user_query = USER_PROMPT_TEMPLATE_DESCRIBE_DATA_FOR_ML.format(dataframe_sample = df_for_prompt, target_column=target_column, model_type=model_type)
        llm_input = [{"role": "user", "content": user_query}]
        llm_response = complete(model=self.llm, prompt=llm_input, options=self.llm_options)
        return df_describe, llm_response