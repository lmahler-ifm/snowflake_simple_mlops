import streamlit as st
import json
from snowflake.cortex import complete, CompleteOptions
from helper_functions.plotting import extract_python_code, extract_json_code
import inspect

# Configure LLM completion options
llm_options = CompleteOptions(
    temperature=0,
    top_p=0
)

# Mapping supported DataFrame types to corresponding icons
dataframe_type_icons = {
    "pandas.core.frame.DataFrame": "🐼",
    "snowflake.snowpark.dataframe.DataFrame": "❄️",
    "snowflake.snowpark.table.Table": "❄️"
}

def test():
    frame = inspect.currentframe().f_back  # Get the caller's frame
    return frame.f_globals

def select_dataframe():
    """
    Identifies supported DataFrame variables available in the global scope.
    Allows the user to select one via Streamlit UI.
    """
    # Retrieve DataFrame variables from the global scope
    available_dataframes = {
        var_name: (var_obj, dataframe_type_icons.get(f"{type(var_obj).__module__}.{type(var_obj).__name__}"))
        for var_name, var_obj in globals().items()
        if f"{type(var_obj).__module__}.{type(var_obj).__name__}" in dataframe_type_icons
    }
    
    # Display warning if no supported DataFrames are found
    if not available_dataframes:
        st.warning("No supported DataFrame variables found.")
        st.stop()
    
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
            sample_data = selected_dataframe.limit(10).to_pandas() if dataframe_type.startswith("snowflake") else selected_dataframe.head(10)
            st.dataframe(sample_data.head(5), use_container_width=True)
            suggest_llm_prompts(sample_data.head(5))
        except Exception as error:
            st.error("Error displaying sample data")
            st.error(error)
    
    return selected_dataframe, dataframe_type

def suggest_llm_prompts(sample_data):
    """
    Generates and displays suggested prompts for analyzing the selected DataFrame.
    """
    with st.form("Prompt Suggestions:", border=False):
        if st.form_submit_button("🤖 What can I ask?"):
            system_prompt = """
            You will later be tasked to create Plotly charts and display them in Streamlit. 
            But first the user wants to understand what kind of questions they can ask.
            For that, you are provided the first 10 rows of a dataframe.

            Make sure to suggest prompts that:
            * Generate analytical insights based on dataframe transformations.
            * Include instructions about how the plot should look.

            Return the suggested prompts as JSON using the following format:
            {'prompt': prompt, 'prompt_explanation': prompt_explanation}
            Only return the JSON, no other content.
            """
            user_prompt = f"""
            I have the following data:
            {sample_data.to_markdown()}
            Suggest 3 prompts that I could use.
            """
            llm_input = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            llm_response = complete(model="mistral-large2", prompt=llm_input, options=llm_options)
            suggested_prompts = json.loads(extract_json_code(llm_response))
            
            #with st.expander("Suggestions:", expanded=True):
            #    for prompt in suggested_prompts:
            #        st.code(prompt['prompt'], language=None)
            for prompt in suggested_prompts:
                st.code(prompt['prompt'], language=None)

def generate_plotly_code(df, dataframe_type):
    """
    Asks the LLM to generate Plotly visualization code based on user input and the selected DataFrame.
    """
    with st.form("Ask LLM"):
        user_query = st.text_area("What can I help you with?", height=4*34)
        
        if st.form_submit_button("🤖 Ask Cortex!"):
            system_prompt = f"""
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
            """
            system_prompt += f"Use df in your code to reference the dataframe. The dataframe has the following columns: {df.columns}"
            
            llm_input = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
            llm_response = complete(model="mistral-large2", prompt=llm_input, options=llm_options)
            
            try:
                generated_code = extract_python_code(llm_response)
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
                retry_response = complete(model="mistral-large2", prompt=llm_input, options=llm_options)
                
                try:
                    retry_code = extract_python_code(retry_response)
                    exec(retry_code)
                    with st.expander("View Adjusted Code"):
                        st.code(retry_code)
                except Exception as retry_error:
                    st.error("Adjusted code also contains errors.")
                    st.error(retry_error)

def get_cortex_helper():
    st.subheader('🤖 Ask Cortex about your Data! ', help='Select a dataframe and ask Cortex for generating plots.')
    dataframe, dataframe_type = select_dataframe()
    generate_plotly_code(dataframe, dataframe_type)