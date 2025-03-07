import streamlit as st
    
def get_snowsight_url(session, link_description, path):
    url = session.sql(f"SELECT CONCAT_WS('/', 'https://app.snowflake.com',CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT_NAME(), '{path}') AS BASE_URL").collect()[0]['BASE_URL']
    return st.info(f'**{link_description}**:\n\n {url}')