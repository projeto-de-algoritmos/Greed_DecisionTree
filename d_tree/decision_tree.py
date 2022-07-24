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

    def sub_tree(self, feature, train_data, classes):
            v_count_dict = train_data[feature].value_counts(sort=False)
            
            tree = {}
            
            for f_value, count in v_count_dict.iteritems():
                f_value_data = train_data[train_data[feature] == f_value]
                
                node = False
                for clss in classes:
                    classes_number = f_value_data[f_value_data[self.label] == clss].shape[0]
                    
                    if classes_number == count:
                        tree[f_value] = clss
                        train_data = train_data[train_data[feature] != f_value]
                        node = True
                        
                if not node:
                    tree[f_value] = "?"
            
            return tree, train_data

    def tree_maker(self, root, previous_feature_value, train_data, classes):
        
        if train_data.shape[0] != 0:
            max_info = _most_info(self, classes, train_data)
            tree, train_data = sub_tree(self, max_info, train_data, classes)
            next_root = None
            
        if previous_feature_value != None:
            root[previous_feature_value] = dict()
            root[previous_feature_value][max_info] = tree
            next_root = root[previous_feature_value][max_info]
        
        else:
            root[max_info] = tree
            next_root = root[max_info]
            
        for node, branch in list(next_root.items()):
            if branch == "?":
                f_value_data = train_data[train_data[max_info] == node]
                tree_maker(self, node, previous_feature_value, f_value_data, classes)

    def id3(self, train_data_m):
        train_data = train_data_m.copy()
        
        tree = {}
        
        classes = train_data[self.label].unique()
        
        tree_maker(self, tree, None, train_data_m, classes)
        
        return tree
    
    def predictions(tree, instance):
        if not isinstance(tree, dict):
            return tree
        else:
            root_node = next(iter(tree))
            f_value = instance[root_node]
            if f_value in tree[root_node]:
                return predictions(tree[root_node], f_value, instance)
            else:
                return None
            