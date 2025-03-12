# Simple MLOps in Snowflake
This repo contains a very simple example for a classical machine learning problem.  
Given some customer data from a fictional Ecommerce Company, we want to predict the yearly spent amount.

## What you'll do
In this example, you'll perform the following steps:


## Requirements
* Snowflake Account

## Get Started
Register for a free Snowflake Trial Account:
- [Free Snowflake Trial Account](https://signup.snowflake.com/)

> [!IMPORTANT]
> Some features like the Feature Store require Snowflake Enterprise Edition or higher. Availability of specific Cortex LLM models can be found [here](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#availability).

Integrate this Github Repository with Snowflake by running the following SQL code in a Snowflake Worksheet:
```sql
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE DATABASE SIMPLE_MLOPS_DEMO;
USE SCHEMA SIMPLE_MLOPS_DEMO.PUBLIC;

-- Create the integration with Github
CREATE OR REPLACE API INTEGRATION GITHUB_INTEGRATION_SNOWFLAKE_SIMPLE_MLOPS
    api_provider = git_https_api
    api_allowed_prefixes = ('https://github.com/michaelgorkow/')
    enabled = true
    comment='Michaels repository containing all the awesome code.';

-- Create the integration with the Github repository
CREATE GIT REPOSITORY GITHUB_REPOSITORY_SNOWFLAKE_SIMPLE_MLOPS
	ORIGIN = 'https://github.com/michaelgorkow/snowflake_simple_mlops' 
	API_INTEGRATION = 'GITHUB_INTEGRATION_SNOWFLAKE_SIMPLE_MLOPS' 
	COMMENT = 'Repository from Michael Gorkow with a simple MLOps Demo.';

-- Fetch most recent files from Github repository
ALTER GIT REPOSITORY GITHUB_REPOSITORY_SNOWFLAKE_SIMPLE_MLOPS FETCH;

-- Run the Setup Script from Github to create demo asset
EXECUTE IMMEDIATE FROM @SIMPLE_MLOPS_DEMO.PUBLIC.GITHUB_REPOSITORY_SNOWFLAKE_SIMPLE_MLOPS/branches/main/_internal/setup.sql;

CREATE OR REPLACE NOTIFICATION INTEGRATION ML_MAIL_INTEGRATION
  TYPE=EMAIL
  ENABLED=TRUE;
```

## Snowflake Features in this demo
* [Snowflake's Git Integration](https://docs.snowflake.com/en/developer-guide/git/git-overview)
* [Snowpark](https://docs.snowflake.com/en/developer-guide/snowpark/python/index)
* [Snowpark ML](https://docs.snowflake.com/en/developer-guide/snowpark-ml/overview)
* [Snowflake Feature Store](https://docs.snowflake.com/en/developer-guide/snowpark-ml/feature-store/overview)
* [Snowflake Model Registry](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)
* [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions)
* [Snowflake Notifications](https://docs.snowflake.com/en/user-guide/notifications/about-notifications)

## API Documentation
* [Snowpark API](https://docs.snowflake.com/developer-guide/snowpark/reference/python/latest/snowpark/index)
* [Snowpark ML API](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/index)
* [Snowflake Feature Store API](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/feature_store)
* [Snowflake Model Registry API](https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/registry)