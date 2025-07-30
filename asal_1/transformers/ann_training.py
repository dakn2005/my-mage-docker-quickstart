import gc
import torch
import torch.optim as optim
import torch.nn as nn
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Tuple, Dict, Union 
from pandas import DataFrame, Series

from asal_1.utils.ann_class import Net

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data: Dict[str, Union[DataFrame, DataFrame, Series, Series]], *args, **kwargs):
    epochs = 100
    learning_rate = 0.01

    ANN_model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ANN_model.parameters(), lr=learning_rate)

    X_train, X_test, y_train, y_test = data["asal_data_transformed"]

    torch.set_default_tensor_type(torch.FloatTensor)

    input_tensor = torch.from_numpy(X_train.values).float()
    target_tensor = torch.from_numpy(y_train.values).float()
    test_target_tensor = torch.from_numpy(y_test.values).float()

    training_losses = []

    return X_train

    for epoch in range(epochs):  # loop over the dataset multiple times
        with torch.no_grad():
            # training set
            outputs = ANN_model(input_tensor)
            loss = criterion(outputs, target_tensor.long())

            # backward pass
            optimizer.zero_grad(set_)
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())

            # print loss at each epoch
            if (epoch % 10 == 0):
                print('Batch 100 Epoch {}: Loss = {:.4f}'.format(epoch+1, loss.item()))
                del outputs
                del loss
                del optimizer
                gc.collect()
            
            

    return ANN_model

# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'
