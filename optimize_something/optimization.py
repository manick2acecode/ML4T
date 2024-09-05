""""""
from matplotlib.dates import MonthLocator, DateFormatter

"""MC1-P2: Optimize a portfolio.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from util import get_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality  		  	   		  		 			  		 			     			  	 
def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1, 0, 0),
        ed=dt.datetime(2009, 1, 1, 0, 0),
        syms=["GOOG", "APPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		  		 			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 			  		 			     			  	 
    statistics.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			     			  	 
    :type sd: datetime  		  	   		  		 			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			     			  	 
    :type ed: datetime  		  	   		  		 			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 			  		 			     			  	 
        symbol in the data directory)  		  	   		  		 			  		 			     			  	 
    :type syms: list  		  	   		  		 			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 			  		 			     			  	 
        code with gen_plot = False.  		  	   		  		 			  		 			     			  	 
    :type gen_plot: bool  		  	   		  		 			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 			  		 			     			  	 
    :rtype: tuple  		  	   		  		 			  		 			     			  	 
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 			  		 			     			  	 
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  		 			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
    prices = prices/prices.iloc[0, :]

    # find the allocations for the optimal portfolio
    stock_len = len(syms)
    # Use a uniform allocation of 1/n to each of the n assets as your initial guess
    allocs = [1.0 / stock_len, ] * stock_len
    # For bounds, pass in seq of 2-tuples (<low>, <high>). supply as many tuples as no. of stocks in your portfolio
    bounds = [(0, 1) for _ in range(stock_len)]
    # constraints = ({ ‘type’: ‘ineq’, ‘fun’: lambda inputs: 50.0 – np.sum(inputs) })
    consts = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    # minimize using scipy
    results = opt.minimize(minimize_sharpe_ratio, x0=allocs, args=(prices,), method="SLSQP", bounds=bounds, constraints=consts)
    allocs = results.x
    print(f"minimize_sharpe_ratio allocs : {allocs}")
    # Get daily portfolio value and other stats
    cum_return, avg_daily_rtn, std_daily_rtn, sharpe_ratio, port_val = assess_portfolio(allocs, prices)

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  		 			  		 			     			  	 
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        df_temp = df_temp / df_temp.iloc[0, :]
        ax = df_temp.plot()
        ax.set_title("Optimized Portfolio and SPY \n", fontsize=18)
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax.margins(x=0)
        ax.set_ylabel('Normalized Price', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        plt.grid()
        plt.show()
        plt.savefig('mmahalingam6_Figure1.png')
        pass

    return allocs, cum_return, avg_daily_rtn, std_daily_rtn, sharpe_ratio


def minimize_sharpe_ratio(allocs, prices):
    # 3rd argument is Sharpe Ratio
    return -1 * assess_portfolio(allocs, prices)[3]


def assess_portfolio(prices, allocs):
    # compute daily portfolio values
    port_val = compute_daily_port_values(prices, allocs)

    # Daily Returns
    daily_returns = compute_daily_returns(port_val)

    # Average Daily Return
    avg_daily_rtn = daily_returns.mean()
    #print(f"avg_daily_rtn: {avg_daily_rtn}")

    # Standard Deviation of Daily Return
    std_daily_rtn = daily_returns.std()
    #print(f"std_daily_rtn: {std_daily_rtn}")

    # Cummulative Return
    cum_return = (port_val[-1] / port_val[0]) - 1
    #print(f"cum_return: {cum_return}")

    # Sharpe Ratio, Risk Adjusted Return, 0%
    sharpe_ratio = (avg_daily_rtn - 0) / std_daily_rtn
    # 252 trading day samples
    sharpe_ratio = np.sqrt(252.0) * sharpe_ratio
    #print(f"sharpe_ratio: {sharpe_ratio}")

    return [cum_return, avg_daily_rtn, std_daily_rtn, sharpe_ratio, port_val]


def compute_daily_port_values(allocs, prices):
    start_val = 1000000.0 # initial investment
    normed_prices = prices / prices.iloc[0]
    allocated = normed_prices * allocs
    pos_val = allocated * start_val
    port_val = pos_val.sum(axis=1)
    return port_val


def compute_daily_returns(df):
    daily_returns = (df/df.shift(1)) - 1
    daily_returns.iloc[0] = 0
    return daily_returns


def test_code():
    """  		  	   		  		 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		  		 			  		 			     			  	 
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio  		  	   		  		 			  		 			     			  	 
    allocations, cum_return, avg_daily_rtn, std_daily_rtn, sharpe_ratio = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Volatility (stdev of daily returns): {std_daily_rtn}")
    print(f"Average Daily Return: {avg_daily_rtn}")
    print(f"Cumulative Return: {cum_return}")


if __name__ == "__main__":
    test_code()
