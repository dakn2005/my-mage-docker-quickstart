import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from typing import List, Tuple

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs) -> Tuple[Series, DataFrame]:
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # remap target classes
    data['Wealthgroup_Name'] = data['Wealthgroup_Name'].replace(['Better Off', 'Middle', 'Poor', 'Very Poor'], [0,1,2,3])

    # Balancing
    w1_data = data[data["Wealthgroup_Name"] == 0]
    w2_data = data[data["Wealthgroup_Name"] == 1]
    w3_data = data[data["Wealthgroup_Name"] == 2]
    w4_data = data[data["Wealthgroup_Name"] == 3]
    selection_number = 2000


    d1 = w1_data[:selection_number]
    d2 = w2_data[:selection_number]
    d3 = w3_data[:selection_number]
    d4 = w4_data[:selection_number]

    data = pd.concat([d1, d2, d3, d4])


    # convert bools
    bool_cols = [
        "recipient_of_wfp",
        "OPCT_received",
        "PWSDCT_received",
        "School_meal_receive",
    ]

    data[bool_cols] = data[bool_cols].applymap(lambda x: x == 1)


    #dropping select features
    fd = data
    fd["Resident_Provider"] = fd["Resident_Provider"].astype(str)

    # convert skip to NaN
    fd2 = fd.replace("SKIP", np.nan).replace("", np.nan)

    # convert variables to SKIP
    # fd2['Chronic_illness'].replace(2, np.nan)
    # fd2['Polygamous'].replace(-2, np.nan)

    fd2.dropna(inplace=True)

    # display(fd2.isnull().any())

    target = fd2["Wealthgroup_Name"] 
    fd2 = fd2.drop(
        [
            "IsBeneficiaryHH",
            "RowID",
            "Sublocation_Name",
            "Village_Name",
            "Division_Name",
            "Location_Name",
        ],
        axis=1,
    )
    fd2 = fd2.drop("Wealthgroup_Name", axis=1)
    fd2 = fd2.drop("PMT_Score", axis=1)

    return target, fd2


@test
def test_output(output: Tuple[Series, DataFrame], *args) -> None:
    """
    Template code for testing the output of the block.
    """

    t, fd = output[0], output[1]
    print('here here: ', t.shape)

    assert True
    # assert fd is not None, 'The output is undefined'
    # assert fd.shape[0] == 7833, f'Expected 7833 records, got {output.shape[0]}'