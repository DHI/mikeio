def safe_length(input_list):
    """
    Get the length of a Python or C# list.

    Usage:
       safe_length(input_list)

    input_list : Python or C# list

    Return:
        int
           Integer giving the length of the input list.
    """
    return getattr(input_list, 'Count', len(input_list))
