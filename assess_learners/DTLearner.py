""""""
"""  		  	   		  		 			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
"""

import numpy as np
import numpy.ma as ma


class DTLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    This is a Decision Tree Linear Regression Learner. It is implemented correctly.
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "mmahalingam6"

    def add_evidence(self, data_x, data_y):
        """  		  	   		  		 			  		 			     			  	 
        Add training data to learner
        :param data_x: A set of feature values used to train the learner  		  	   		  		 			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """
        new_data_x = np.hstack((data_x, data_y.reshape(-1, 1)))
        self.tree = self.build_tree(new_data_x)

    def build_tree(self, data):
        data_y = data[:, -1]
        if data.shape[0] <= self.leaf_size or len(data.shape) == 1:
            return np.array([['leaf', np.mean(data_y), -1, -1]])
        elif np.all(data_y == data[0, -1]):
            return np.array([['leaf', data[0, -1], -1, -1]])
        else:

            high_corr_i = 0
            highest_correlation = -1
            for i in range(data.shape[1] - 1):
                correlation = ma.corrcoef(ma.masked_invalid(data[:, i]), ma.masked_invalid(data_y))[0, 1]
                correlation = abs(correlation)
                if correlation > highest_correlation:
                    highest_correlation = correlation
                    high_corr_i = i
            split_value = np.median(data[:, high_corr_i], axis=0)
            if split_value == max(data[:, high_corr_i]):
                return np.array([['leaf', np.mean(data_y), -1, -1]])

            left_tree = self.build_tree(data[data[:, high_corr_i] <= split_value])
            right_tree = self.build_tree(data[data[:, high_corr_i] > split_value])
            root = np.array([[high_corr_i, split_value, 1, left_tree.shape[0] + 1]])
            decision_tree = np.vstack((np.vstack((root, left_tree)), right_tree))
            return decision_tree

    def query(self, points):
        """  		  	   		  		 			  		 			     			  	 
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  		 			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """
        results = []
        root = self.tree # decision tree
        for i in range(points.shape[0]):
            node = 0
            while root[node, 0] != 'leaf':
                index = root[node, 0]
                split_val = root[node, 1]
                if points[i, int(float(index))] <= float(split_val):
                    left = int(float(root[node, 2]))
                    node = node + left
                else:
                    right = int(float(root[node, 3]))
                    node = node + right
            result = root[node, 1]
            results.append(float(result))
        return np.array(results)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")


