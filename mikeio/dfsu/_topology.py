from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mikecore.DfsuFile import DfsuFile
from mikecore.MeshFile import MeshFile


@dataclass
class ElementTable:
    connectivity: list[NDArray[np.int32]]
    ids: NDArray[np.int32]


@dataclass
class NodeTable:
    coordinates: NDArray[np.float64]
    codes: NDArray[np.int32]
    ids: NDArray[np.int32]


def get_elements_from_source(source: DfsuFile | MeshFile) -> ElementTable:
    # Dfsu element table is a np.array(,dtype=object), while Mesh uses list[np.ndarray], very similar, but not identical
    element_table = get_element_table_from_mikecore(source.ElementTable)
    element_ids = source.ElementIds - 1
    return ElementTable(element_table, element_ids)


def get_nodes_from_source(source: DfsuFile | MeshFile) -> NodeTable:
    xn = source.X
    yn = source.Y
    zn = source.Z
    nc = np.column_stack([xn, yn, zn])
    codes = source.Code
    node_ids = source.NodeIds - 1
    return NodeTable(nc, codes, node_ids)


def get_element_table_from_mikecore(
    element_table: list[np.ndarray],
) -> list[np.ndarray]:
    new_elem_table = element_table.copy()
    for j in range(len(element_table)):
        new_elem_table[j] = element_table[j] - 1
    return new_elem_table
