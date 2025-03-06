import re

# Function that extracts the actual Python code returned by mistral
def extract_python_code(text):
    # Regular expression pattern to extract content between triple backticks with 'python' as language identifier
    pattern = r"```python(.*?)```"

    # re.DOTALL allows the dot (.) to match newlines as well
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Return the matched group, stripping any leading or trailing whitespace
        return match.group(1).strip()
    else:
        return "No Python code found in the input string."
    
# Function to extract JSON code from a string using regex
def extract_json_code(response_text):
    pattern = r"```json(.*?)```"  # Matches content enclosed in triple backticks labeled 'json'
    match = re.search(pattern, response_text, re.DOTALL)  # DOTALL ensures newlines are captured
    return match.group(1).strip() if match else "No JSON code found in the input string."