from typing import Union
from mikecore.DfsFile import DataValueType
from .exceptions import InvalidDataValueType

# TODO relocate this to some more appropriately named file
def to_datatype(datatype: Union[str, int, DataValueType]) -> DataValueType:
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

    if isinstance(datatype, str):
        if datatype not in string_datatype_mapping.keys():
            raise InvalidDataValueType

        return string_datatype_mapping[datatype]

    if isinstance(datatype, int):
        if datatype not in string_datatype_mapping.keys():
            raise InvalidDataValueType

        return string_datatype_mapping[datatype]

    if not isinstance(DataValueType):
        raise ValueError("Data value type not supported")

    return datatype
