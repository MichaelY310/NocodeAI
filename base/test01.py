def a():
    function_definition = """
global dynamic_function
def dynamic_function(name):
    print(f"Hello, I am {name}.")
    """

    exec(function_definition)


a()
dynamic_function("Dynamic Function")