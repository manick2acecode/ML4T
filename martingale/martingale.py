""""""
import matplotlib.pyplot as plt

"""Assess a betting strategy.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
  		  	   		  		 			  		 			     			  	 
Student Name: Manick Mahalingam (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: mmahalingam6 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903847991 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""

import numpy as np


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mmahalingam6"


def gtid():
    """  		  	   		  		 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		  		 			  		 			     			  	 
    :rtype: int  		  	   		  		 			  		 			     			  	 
    """
    return 903847991


def get_spin_result(win_prob):
    """  		  	   		  		 			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		  		 			  		 			     			  	 
    :type win_prob: float  		  	   		  		 			  		 			     			  	 
    :return: The result of the spin.  		  	   		  		 			  		 			     			  	 
    :rtype: bool  		  	   		  		 			  		 			     			  	 
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def gambling_simulator(win_prob, max_bank_roll_val):
    # define winning array for 1000 spins, numpy treats array till 1000 for 1001
    winnings_array = np.full(1001, 80)
    episode_winnings = 0
    temp_inc = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            if temp_inc >= 1001:
                return winnings_array
            winnings_array[temp_inc] = episode_winnings
            temp_inc += 1
            won = get_spin_result(win_prob)
            if won:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount *= 2

                if max_bank_roll_val is not None:

                    if episode_winnings == -max_bank_roll_val:
                        winnings_array[temp_inc:] = episode_winnings
                        return winnings_array

                    if episode_winnings - bet_amount <= -max_bank_roll_val:
                        bet_amount = max_bank_roll_val + episode_winnings

    return winnings_array


def output_figure1(win_prob):
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 1 - 10 episodes, unlimited bankroll")
    plt.xlabel("Number of Bets")
    plt.ylabel("Total Winnings")

    for index in range(10):
        current_episode = gambling_simulator(win_prob, None)
        plt.plot(current_episode)

    plt.savefig(author() + "_figure1.png")
    plt.clf()


def output_figure2_figure3(win_prob):
    file_name = author()
    winnings_array = np.zeros((1000, 1001))
    for index in range(1000):
        curr_episode = gambling_simulator(win_prob, None)
        winnings_array[index] = curr_episode

    std = np.std(winnings_array, axis=0)

    mean_array = np.mean(winnings_array, axis=0)
    util_plot_graph("Figure 2 - Mean, no bankroll limit (1000 episodes)", file_name + "_figure2.png", "mean",
                    mean_array, std)

    median_array = np.median(winnings_array, axis=0)
    util_plot_graph("Figure 3 - Median, no bankroll limit (1000 episodes)", file_name + "_figure3.png", "median",
                    median_array, std)


def output_figure4_figure5(win_prob, max_bank_roll_val):
    file_name = author()
    winnings_array = np.zeros((1000, 1001))
    for index in range(1000):
        curr_episode = gambling_simulator(win_prob, max_bank_roll_val)
        winnings_array[index] = curr_episode

    std = np.std(winnings_array, axis=0)

    mean_array = np.mean(winnings_array, axis=0)
    util_plot_graph("Figure 4 - Mean with bankroll limit $256 (1000 episodes)", file_name + "_figure4.png", "mean",
                    mean_array, std)

    median_array = np.median(winnings_array, axis=0)
    util_plot_graph("Figure 5 - Median with bankroll limit $256 (1000 episodes)", file_name + "_figure5.png", "median",
                    median_array, std)


def util_plot_graph(title, figure_name, prefix, exp_array, exp_std):
    """
        utility method for plotting the graph with mean, median 	  	   		  		 			  		 			     			  	 
    """
    plus_array = exp_array + exp_std
    minus_array = exp_array - exp_std

    plt.axis([0, 300, -256, 100])
    plt.title(title)
    plt.xlabel("Number of Bets")
    plt.ylabel("Total Winnings")

    plt.plot(exp_array, label=prefix)
    plt.plot(plus_array, label=prefix + "_plus_std")
    plt.plot(minus_array, label=prefix + "_minus_std")

    plt.legend()
    plt.savefig(figure_name)
    plt.clf()


def test_code():
    win_prob = 0.474  # set appropriately to the probability of a win
    np.random.seed(gtid())
    max_bank_roll_val = 256
    # print(get_spin_result(win_prob))  # test the roulette spin
    output_figure1(win_prob)
    output_figure2_figure3(win_prob)
    output_figure4_figure5(win_prob, max_bank_roll_val)


if __name__ == "__main__":
    test_code()
