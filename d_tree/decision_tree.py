import pandas as pd
import numpy as np


class DecisionTree:
    """
    class that runs ID3 algorithm, wich uses entropy and total information to train and predict
    """

    def __init__(self, train_data, label):
        self.train_data = train_data
        self.label = label

    def _dataset_entropy(self, classes, train_data):
        total_entropy = 0
        for clss in classes:
            classes_number = train_data[train_data[self.label] == clss].shape[0]
            class_entropy = 0
            if classes_number != 0:
                class_entropy = -(classes_number/train_data.shape[0]) * np.log2(classes_number/train_data.shape[0])
            entropy += class_entropy
        return total_entropy

    def _feature_entropy(self, feature_data, classes):
        feature_entropy = 0
        for clss in classes:
            classes_number = feature_data[feature_data[self.label] == clss].shape[0]
            class_entropy = 0
            if classes_number != 0:
                class_entropy = -(classes_number/feature_data.shape[0]) * np.log2(classes_number/feature_data.shape[0])
            feature_entropy += class_entropy
        return feature_entropy
