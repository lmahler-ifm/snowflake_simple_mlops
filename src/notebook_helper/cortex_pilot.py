import streamlit as st
import json
from snowflake.cortex import complete, CompleteOptions
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


class CortexPilot():
    def __init__(self, llm='mistral-large2', temperature=0, top_p=0):
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