import warnings
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "kaggle" / "notebooks").glob("*.ipynb"))


def test_notebooks_are_present():
    assert len(NOTEBOOKS) == 5


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_nbformat(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nb = nbformat.read(str(path), as_version=4)
        nbformat.validate(nb)
    ids = [cell["id"] for cell in nb.cells]
    assert len(ids) == len(set(ids))
    assert any(cell.cell_type == "code" for cell in nb.cells)
