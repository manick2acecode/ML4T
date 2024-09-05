""""""
"""  		  	   		  		 			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
  		  	   		  		 			  		 			     			  	 
Student Name: Manick Mahalingam		  	   		  		 			  		 			     			  	 
GT User ID: mmahalingam6	  	   		  		 			  		 			     			  	 
GT ID: 903847991 		  	   		  		 			  		 			     			  	 
"""

import random as rand
import time
import numpy as np


class QLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    This is a Q learner object.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		  		 			  		 			     			  	 
    :type num_states: int  		  	   		  		 			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		  		 			  		 			     			  	 
    :type num_actions: int  		  	   		  		 			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type alpha: float  		  	   		  		 			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type gamma: float  		  	   		  		 			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type rar: float  		  	   		  		 			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 			  		 			     			  	 
    :type radr: float  		  	   		  		 			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 			  		 			     			  	 
    :type dyna: int  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_learn = np.zeros((self.num_states, self.num_actions))
        self.experience_t = []

        if self.dyna != 0:
            self.tranz = np.zeros((self.num_states, self.num_actions))
            self.reward = np.zeros((self.num_states, self.num_actions))
            # init tranz count
            self.tranz_c = np.ndarray(shape=(self.num_states, self.num_actions, self.num_states))
            self.tranz_c.fill(0.00001)

    def querysetstate(self, s):
        """  		  	   		  		 			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param s: The new state  		  	   		  		 			  		 			     			  	 
        :type s: int  		  	   		  		 			  		 			     			  	 
        :return: The selected action  		  	   		  		 			  		 			     			  	 
        :rtype: int  		  	   		  		 			  		 			     			  	 
        """
        # random action probability, referenced from navigation project lecture
        # if random is less than random action rate, then choose random action
        # else get max value from s
        self.s = s
        self.a = action = self.get_rand_action(s)
        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action

    def query(self, s_prime, r):
        """  		  	   		  		 			  		 			     			  	 
        Update the Q table and return an action  		  	   		  		 			  		 			     			  	 
  		:param s_prime: The new state
        :type s_prime: int  		  	   		  		 			  		 			     			  	 
        :param r: The immediate reward  		  	   		  		 			  		 			     			  	 
        :type r: float  		  	   		  		 			  		 			     			  	 
        :return: The selected action  		  	   		  		 			  		 			     			  	 
        :rtype: int  		  	   		  		 			  		 			     			  	 
        """

        # Q-table formula update based on the lecture videos in 03-06
        self.q_learn[self.s, self.a] = self.q_learn_formula(self.s, self.a, s_prime, r)

        # random action probability, referenced from navigation project lecture
        # if random is less than random action rate, then choose random action
        # else get max value from prime
        action = self.get_rand_action(s_prime)

        # update experience tuple for each epoch as per lecture video (before dyna)
        self.experience_t.append([self.s, self.a])

        # dyna-q algorithm implementation
        if self.dyna != 0:
            self.implement_dyna_algorithm(s_prime, r)

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        # random action decay rate, after each update, rar = rar * radr
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action
        return action

    def get_rand_action(self, s_prime):
        if rand.uniform(0.0, 1.0) <= self.rar:
            return rand.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_learn[s_prime]);

    def q_learn_formula(self, s, a, s_prime, r):
        # Q-table formula update based on the lecture videos in 03-06
        # Q[s,a] = (1-alpha) * Q[s,a] + alpha * (r + (gamma *Q[s_prime, np.argmax(Q[s_prime, ])])
        return (1 - self.alpha) * self.q_learn[s, a] + self.alpha * (
                r + self.gamma * self.q_learn[s_prime, np.argmax(self.q_learn[s_prime])])

    def implement_dyna_algorithm(self, s_prime, r):
        # increment tcount by 1
        self.tranz_c[self.s, self.a, s_prime] = self.tranz_c[self.s, self.a, s_prime] + 1
        # evaluate T
        self.tranz = self.tranz_c / np.sum(self.tranz_c, axis=2, keepdims=True)
        # R[s, a] = ((1 - alpha) * R[s, a]) + (alpha * r) based on lecture video
        self.reward[self.s, self.a] = ((1 - self.alpha) * self.reward[self.s, self.a]) + (self.alpha * r)

        dyna_tuple_idx = np.random.randint(len(self.experience_t), size=self.dyna)
        for i in dyna_tuple_idx:
            # random state (s), random action (action) based on lecture video
            dyna_s, dyna_action = self.experience_t[i]
            # infer s_prime from T[]
            dyna_s_prime = np.argmax(self.tranz[dyna_s, dyna_action])
            # r = R[s,a]
            dyna_r = self.reward[dyna_s, dyna_action]
            # Update the Q-table in Dyna function
            self.q_learn[dyna_s, dyna_action] = self.q_learn_formula(dyna_s, dyna_action, dyna_s_prime, dyna_r)

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "mmahalingam6"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
