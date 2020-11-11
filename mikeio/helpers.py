from DHI.Generic.MikeZero.DFS import DataValueType

from mikeio.custom_exceptions import InvalidDataValueType


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

    try:
        n = input_list.Count
    except:
        n = len(input_list)

    return n


def to_datatype(datatype_str):
    string_datatype_mapping = {
        "Instantaneous": DataValueType.Instantaneous,
        "Accumulated": DataValueType.Accumulated,
        "StepAccumulated": DataValueType.StepAccumulated,
        "MeanStepBackward": DataValueType.MeanStepBackward,
        "MeanStepForward": DataValueType.MeanStepForward,
        0: DataValueType.Instantaneous,
        1: DataValueType.Accumulated,
        2: DataValueType.StepAccumulated,
        3: DataValueType.MeanStepBackward,
        4: DataValueType.MeanStepForward,
    }
    if datatype_str not in string_datatype_mapping.keys():
        raise InvalidDataValueType

    return string_datatype_mapping[datatype_str]
