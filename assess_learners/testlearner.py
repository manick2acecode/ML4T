""""""
"""  		  	   		  		 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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

import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import DTLearner as dtlearn
import RTLearner as rtlearn
import BagLearner as baglearn


def plot_image(in_sample, out_sample, title, xlabel, xlim_2, ylabel, ylim_2, legend, image_name):
    plt.plot(in_sample)
    plt.plot(out_sample)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xlim(1, xlim_2)
    plt.ylabel(ylabel)
    plt.ylim(0, ylim_2)
    plt.legend(legend)
    plt.grid()
    plt.savefig(image_name)
    plt.close("all")


def question_1(train_x_prm, train_y_prm, test_x_prm, test_y_prm, leaf_size_prm):
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, leaf_size_prm+1):
        # create a learner and train it
        learner_tmp = dtlearn.DTLearner(leaf_size=i, verbose=False)
        learner_tmp.add_evidence(train_x_prm, train_y_prm)
        # evaluate in sample, training data
        pred_y_train = learner_tmp.query(train_x_prm)  # get the predictions
        rmse_in = math.sqrt(((train_y_prm - pred_y_train) ** 2).sum() / train_y_prm.shape[0])
        rmse_in_sample.append(rmse_in)
        # evaluate out of sample, test data
        pred_y_test = learner_tmp.query(test_x_prm)  # get the predictions
        rmse_out = math.sqrt(((test_y_prm - pred_y_test) ** 2).sum() / test_y_prm.shape[0])
        rmse_out_sample.append(rmse_out)
    # plot the image
    plot_image(rmse_in_sample, rmse_out_sample, "DTLearner - RMSE vs Leaf Size", "Leaf Size", leaf_size_prm, "RMSE", 0.01, ["In-Sample", "Out-Sample"], "Question-1.png")


def question_2(train_x_prm, train_y_prm, test_x_prm, test_y_prm, leaf_size_prm):
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, leaf_size_prm+1):
        # create a learner and train it
        learner_tmp = baglearn.BagLearner(learner=dtlearn.DTLearner, kwargs={"leaf_size": i}, bags=10, boost=False,
                                          verbose=True)
        learner_tmp.add_evidence(train_x_prm, train_y_prm)
        # evaluate in sample, training data
        pred_y_train = learner_tmp.query(train_x_prm)  # get the predictions
        rmse_in = math.sqrt(((train_y_prm - pred_y_train) ** 2).sum() / train_y_prm.shape[0])
        rmse_in_sample.append(rmse_in)
        # evaluate out of sample, test data
        pred_y_test = learner_tmp.query(test_x_prm)  # get the predictions
        rmse_out = math.sqrt(((test_y_prm - pred_y_test) ** 2).sum() / test_y_prm.shape[0])
        rmse_out_sample.append(rmse_out)
    # plot the image
    plot_image(rmse_in_sample, rmse_out_sample, "BagLearner - RMSE vs Leaf Size (10 Bags)", "Leaf Size", leaf_size_prm, "RMSE", 0.01, ["In-Sample", "Out-Sample"], "Question-2.png")


def question_3_i(train_x_prm, train_y_prm, leaf_size_prm):
    time_dtlearner = []
    time_rtlearner = []
    # DT Learner Time to run calculation
    for i in range(1, leaf_size_prm+1):
        start_time = time.time()
        learner_dt = dtlearn.DTLearner(leaf_size=i, verbose=True)
        learner_dt.add_evidence(train_x_prm, train_y_prm)
        end_time = time.time()
        time_dtlearner.append(end_time - start_time)
    # RT Learner Time to run calculation
    for i in range(1, leaf_size_prm+1):
        start_time = time.time()
        learner_rt = rtlearn.RTLearner(leaf_size=i, verbose=True)
        learner_rt.add_evidence(train_x_prm, train_y_prm)
        end_time = time.time()
        time_rtlearner.append(end_time - start_time)
    # plot the image
    plot_image(time_dtlearner, time_rtlearner, "Training Time - DTLearner vs RTLearner", "Leaf Size", leaf_size_prm, "Time (milliseconds)", 0.25, ["DTLearner", "RTLearner"], "Question-3-i.png")


def question_3_ii(train_x_prm, train_y_prm, leaf_size_prm):
    mae_dtlearner = []
    mae_rtlearner = []
    # Mean Arthimetic Error calculation
    for i in range(1, leaf_size_prm+1):
        # Mean Arithmetic Error for DTLearner
        learner_dt = dtlearn.DTLearner(leaf_size=i, verbose=True)
        learner_dt.add_evidence(train_x_prm, train_y_prm)
        pred_y_dt = learner_dt.query(train_x_prm)
        pred_y_dt = np.array(pred_y_dt)
        train_y_dt = np.array(train_y_prm)
        mae_dt = np.mean(np.abs(train_y_dt - pred_y_dt)) * 100
        mae_dtlearner.append(mae_dt)
        # Mean Arithmetic Error for RTLearner
        learner_rt = rtlearn.RTLearner(leaf_size=i, verbose=True)
        learner_rt.add_evidence(train_x_prm, train_y_prm)
        pred_y_rt = learner_rt.query(train_x_prm)
        pred_y_rt = np.array(pred_y_rt)
        train_y_rt = np.array(train_y_prm)
        mae_rt = np.mean(np.abs(train_y_rt - pred_y_rt)) * 100
        mae_rtlearner.append(mae_rt)
    # plot the image
    plot_image(mae_dtlearner, mae_rtlearner, "MAE - DTLearner vs RTLearner", "Leaf Size", leaf_size_prm, "Mean Absolute Error", 0.8, ["DTLearner", "RTLearner"], "Question-3-ii.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    data = np.genfromtxt(sys.argv[1], delimiter=",")  # Referred from grade_learners.py
    data = data[1:, 1:]  # exclude header and date (1st column)

    print(f"{data}")

    # set global leaf_size
    leaf_size_def = 40

    # compute how much of the data is training and testing  		  	   		  		 			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    print(f"train_rows \n {train_rows}")

    print(f"test_rows \n {test_rows}")
    # separate out training and testing data  		  	   		  		 			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"train_x \n {train_x}")

    print(f"train_y \n {train_y}")

    print(f"test_x \n {test_x}")

    print(f"test_y \n {test_y}")

    # Question 1, DTLearner - RMSE vs Leaf Size (Root Mean Square Deviation)
    question_1(train_x, train_y, test_x, test_y, leaf_size_def)

    # Question 2, BagLearner(10 Bags) - RMSE vs Leaf Size (Root Mean Square Deviation)
    question_2(train_x, train_y, test_x, test_y, leaf_size_def)

    # Question 3.i, Training Time - DTLearner vs RTLearner
    question_3_i(train_x, train_y, leaf_size_def)

    # Question 3.ii, MAE - DTLearner vs RTLearner
    question_3_ii(train_x, train_y, leaf_size_def)
