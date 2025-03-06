import inspect

def get_function_source_recursively(func, max_depth=1, current_depth=0, seen=None):
    """
    Recursively retrieves the source code of a function and its dependencies 
    up to a given depth.
    
    :param func: The function to retrieve the source for.
    :param max_depth: Maximum recursion depth for called functions.
    :param current_depth: The current recursion level.
    :param seen: A set to track already processed functions (to avoid duplication).
    :return: The concatenated source code of the function and its dependencies.
    """
    if seen is None:
        seen = set()

    if func in seen or current_depth > max_depth:
        return ""  # Stop if max depth is reached or function is already processed
    
    seen.add(func)
    
    try:
        source = inspect.getsource(func)
    except OSError:
        return f"# Source code not available for {func.__name__}\n"
    
    if current_depth < max_depth:
        # Find functions called inside this function
        called_functions = set()
        for name, obj in func.__globals__.items():
            if inspect.isfunction(obj) and name in source:
                called_functions.add(obj)
        
        # Recursively get the source of called functions
        called_sources = "\n".join(
            get_function_source_recursively(f, max_depth, current_depth + 1, seen) 
            for f in called_functions
        )
        
        return source + "\n" + called_sources
    else:
        return source  # Stop recursion when max_depth is reached