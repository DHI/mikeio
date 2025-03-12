from __future__ import annotations
from pathlib import Path
from ._pfsdocument import PfsDocument
from ._pfssection import PfsNonUniqueList, PfsSection


def read_pfs(
    filename: str | Path,
    encoding: str = "cp1252",
    unique_keywords: bool = False,
) -> PfsDocument:
    """Read a pfs file to a Pfs object for further analysis/manipulation.

    Parameters
    ----------
    filename: str or Path
        File name including full path to the pfs file.
    encoding: str, optional
        How is the pfs file encoded? By default 'cp1252'
    unique_keywords: bool, optional
        Should the keywords in a section be unique? Some tools e.g. the
        MIKE Plot Composer allows non-unique keywords.
        If True: warnings will be issued if non-unique keywords
        are present and the first occurence will be used
        by default False

    Returns
    -------
    PfsDocument
        A PfsDocument object

    """
    return PfsDocument(filename, encoding=encoding, unique_keywords=unique_keywords)


__all__ = [
    "Pfs",
    "PfsDocument",
    "PfsNonUniqueList",
    "PfsSection",
    "read_pfs",
]
