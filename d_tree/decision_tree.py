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
            total_entropy += class_entropy
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
    
    def _information_gain_per_feature(self, feature, classes, train_data):
        f_list = train_data[feature].unique()
        feature_information = .0
        train_data_size = train_data.shape[0]

        for f in f_list:
            f_value_data = train_data[train_data[feature] == f]
            f_value_count = f_value_data.shape[0]
            f_value_entropy = self._feature_entropy(f_value_data, classes)
            feature_information += (f_value_count /
                                    train_data_size) * f_value_entropy

        entropy = self._dataset_entropy(classes, train_data)
        return entropy - feature_information

    def _most_info(self, classes, train_data):
        f_list = train_data.columns.drop(self.label)
        max_info = -1
        max_feature = None

        for f in f_list:
            f_gain = self._information_gain_per_feature(f, classes, train_data)
            if max_info < f_gain:
                max_info = f_gain
                max_feature = f
        return max_feature, max_info
