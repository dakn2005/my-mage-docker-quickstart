from mage_ai.io.file import FileIO
from pandas import DataFrame, Series
from typing import List, Tuple

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data: Tuple[Series, DataFrame], *args, **kwargs) -> Tuple[DataFrame, DataFrame, Series, Series]:
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    # FileIO().export(data, "asal_transformed.csv")

    return data


@test
def test_output(output: Tuple[Series, DataFrame], *args) -> None:
    """
    Template code for testing the output of the block.
    """

    # print(output)
    assert output is not None, f'Expected 7833 records, got {output.shape[0]}'


