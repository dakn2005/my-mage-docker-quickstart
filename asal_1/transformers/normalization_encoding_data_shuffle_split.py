from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series
from typing import List, Tuple, Dict, Union


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def normalization(df):
    scaler = MinMaxScaler()  # default=(0, 1)

    numerical = [
        "Age",
        "Polygamous",
        "Children_Under_15_outside_settlement",
        "Kids_Under_15_In_Settlement",
        "Spouses_on_settlement",
        "Spouses_Outside_HH",
        "Donkeys_owned",
        "Camels_owned",
        "Zebu_cattle_owned",
        "Shoats_owned",
        "Nets_owned",
        "Hooks_owned",
        "Boats_rafts_owned",
        # 'PMT_Score',
    ]

    features_minmax_transform = pd.DataFrame(data=df)
    # display(features_log_minmax_transform[:1])
    features_minmax_transform[numerical] = scaler.fit_transform(fd2[numerical])

    return features_minmax_transform

def one_hot_encoding(df):
    return pd.get_dummies(df)


@transformer
def transform(data: Dict[str, Union[Series, DataFrame]], *args, **kwargs) -> Tuple[DataFrame, DataFrame, Series, Series]:
    return data

    target, df = data['asal_data_transformed']

    normalized_df = normalization(d)
    features_final = one_hot_encoding(normalized_df)

    X_train, X_test, y_train, y_test = train_test_split(
        features_final, target, test_size=0.2, random_state=0
    )
   
    print(
        X_train.shape,
        X_test.shape
    )

    return X_train, X_test, y_train, y_test



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    # print(output)

    assert output is not None, 'The output is undefined'
    # assert df.shape[0] > 0, 'must have some data'
