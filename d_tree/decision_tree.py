from cgi import test
import re
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

    def _generate_sub_tree(self, feature, classes, train_data):
        f_value_dict = train_data[feature].value_counts(sort=False)
        tree = {}

        for f, count in f_value_dict.iteritems():
            f_value_data = train_data[train_data[feature] == f]
            is_pure_class = False

            for clss in classes:
                class_numbers = f_value_data[f_value_data[self.label]
                                             == clss].shape[0]
                if class_numbers == count:
                    tree[f] = clss
                    train_data = train_data[train_data[feature] != f]
                    is_pure_class = True
            if not is_pure_class:
                tree[f] = '.'
        return tree

    def _create_tree(self, root, prev_feature, updated_dataset, classes):
        if updated_dataset.shape[0] != 0:
            max_feature, _ = self._most_info(classes, updated_dataset)
            tree = self._generate_sub_tree(
                max_feature, classes, updated_dataset)
            next_root = None

            if prev_feature != None:
                root[prev_feature] = {}
                root[prev_feature][max_feature] = tree
                next_root = root[prev_feature][max_feature]
            else:
                root[max_feature] = tree
                next_root = root[max_feature]

            for node, branch in list(next_root.items()):
                if branch == '.':
                    f_value_data = updated_dataset[updated_dataset[max_feature] == node]
                    self._create_tree(next_root, node, f_value_data, classes)

    def id3(self):
        tree = {}
        dataset = self.train_data.copy()
        classes = dataset[self.label].unique()
        self._create_tree(tree, None, dataset, classes)
        return tree

    def predict(self, tree, instance):
        if not isinstance(tree, dict):
            return tree
        else:
            root = next(iter(tree))
            f_value = instance[root]
            if f_value in tree[root]:
                return self.predict(tree[root][f_value], instance)
            else:
                return None

    def evaluate(self, tree, test_data):
        correct = 0
        wrong = 0
        for _, row in test_data.iterrows():
            result = self.predict(tree, row)
            if result == row[self.label]:
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong)