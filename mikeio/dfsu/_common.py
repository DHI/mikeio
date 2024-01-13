from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mikecore.DfsuFile import DfsuFile
from mikecore.MeshFile import MeshFile


# TODO consider where to put these two classes
@dataclass
class ElementTable:
    connectivity: list[NDArray[np.int32]]
    ids: NDArray[np.int32]

    # def __getitem__(self, key: int) -> tuple[NDArray[np.int32], np.int32]:
    #    return self.connectivity[key], self.ids[key]


@dataclass
class NodeTable:
    coordinates: NDArray[np.float64]
    codes: NDArray[np.int32]
    ids: NDArray[np.int32]

    # def __getitem__(self, key: int) -> tuple[NDArray[np.float64], np.int32, np.int32]:
    #    return self.coordinates[key], self.codes[key], self.ids[key]


def get_elements_from_source(source: DfsuFile | MeshFile) -> ElementTable:
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


def _offset_element_table_by(
    element_table: list[np.ndarray], *, offset: int
) -> list[np.ndarray]:
    new_elem_table = element_table.copy()
    for j in range(len(element_table)):
        new_elem_table[j] = element_table[j] + offset
    return new_elem_table


def get_element_table_from_mikecore(
    element_table: list[np.ndarray],
) -> list[np.ndarray]:
    return _offset_element_table_by(element_table, offset=-1)


def element_table_to_mikecore(element_table: list[np.ndarray]) -> list[np.ndarray]:
    return _offset_element_table_by(element_table, offset=1)
