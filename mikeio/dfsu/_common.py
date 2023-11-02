from __future__ import annotations

import numpy as np

from mikecore.DfsuFile import DfsuFile
from mikecore.MeshFile import MeshFile

def _get_elements_from_source(source: DfsuFile | MeshFile) -> tuple[np.ndarray, np.ndarray]:
    element_table = _get_element_table_from_mikecore(
        source.ElementTable
    )
    element_ids = source.ElementIds - 1
    return element_table, element_ids


def _get_nodes_from_source(source: DfsuFile | MeshFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xn = source.X
    yn = source.Y
    zn = source.Z
    nc = np.column_stack([xn, yn, zn])
    codes = np.array(list(source.Code))
    node_ids = source.NodeIds - 1
    return nc, codes, node_ids

def _offset_element_table_by(element_table:np.ndarray, *, offset:int) -> np.ndarray:
    offset = int(offset)
    new_elem_table = element_table.copy()
    for j in range(len(element_table)):
        new_elem_table[j] = element_table[j] + offset
    return new_elem_table

    
def _get_element_table_from_mikecore(element_table:np.ndarray) -> np.ndarray:
    return _offset_element_table_by(element_table, offset=-1)

def _element_table_to_mikecore(element_table:np.ndarray) -> np.ndarray:
    return _offset_element_table_by(element_table, offset=1)
