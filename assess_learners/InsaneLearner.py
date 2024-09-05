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
import LinRegLearner as linear
import BagLearner as bag


class InsaneLearner(object):
    """
    This is a Insane Linear Regression Learner. It is implemented correctly.
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bags = 20
        self.insane_learners = []
        print("inside insane learner")
        for i in range(self.bags):
            self.learners.append(
                bag.BagLearner(learner=linear.LinRegLearner, kwargs={}, bags=self.bags, boost=False, verbose=False))

    def author(self):
        return "mmahalingam6"

    def add_evidence(self, data_x, data_y):
        for insane_learner in self.insane_learners:
            insane_learner.add_evidence(data_x, data_y)

    def query(self, points):
        insane_output = []
        for in_learner in self.insane_learners:
            insane_output.append(in_learner.query(points))
        insane_output = np.mean(np.array(insane_output), axis=0)
        return insane_output


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
