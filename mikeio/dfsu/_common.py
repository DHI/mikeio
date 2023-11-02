from __future__ import annotations

import numpy as np

from mikecore.DfsuFile import DfsuFile
from mikecore.MeshFile import MeshFile

def get_elements_from_source(source: DfsuFile | MeshFile) -> tuple[list[np.ndarray], np.ndarray]:
    element_table = get_element_table_from_mikecore(
        source.ElementTable
    )
    element_ids = source.ElementIds - 1
    return element_table, element_ids


def get_nodes_from_source(source: DfsuFile | MeshFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xn = source.X
    yn = source.Y
    zn = source.Z
    nc = np.column_stack([xn, yn, zn])
    codes = source.Code
    node_ids = source.NodeIds - 1
    return nc, codes, node_ids

def _offset_element_table_by(element_table: list[np.ndarray], *, offset:int) -> list[np.ndarray]:
    new_elem_table = element_table.copy()
    for j in range(len(element_table)):
        new_elem_table[j] = element_table[j] + offset
    return new_elem_table
    
def get_element_table_from_mikecore(element_table: list[np.ndarray]) -> list[np.ndarray]:
    return _offset_element_table_by(element_table, offset=-1)

def element_table_to_mikecore(element_table: list[np.ndarray]) -> list[np.ndarray]:
    return _offset_element_table_by(element_table, offset=1)