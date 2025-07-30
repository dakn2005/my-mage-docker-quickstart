from xgboost import XGBClassifier
from pandas import DataFrame, Series
from typing import List, Tuple, Dict, Union 


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data: Dict[str, Union[DataFrame, DataFrame, Series, Series]], *args, **kwargs) -> XGBClassifier:
    
    X_train, X_test, y_train, y_test = data["asal_data_transformed"]

    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    return clf 


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
