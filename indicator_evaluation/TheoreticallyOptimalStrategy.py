""""""

"""MC2-P1: Market simulator.  		  	   		  		 			  		 			     			  	 

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
GT User ID: mmahalingam  		  	   		  		 			  		 			     			  	 
GT ID: 903847991	  	   		  		 			  		 			     			  	 
"""

import numpy as np
import datetime as dt
import pandas as pd
from util import get_data
from marketsimcode import compute_portvals
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter


class TheoreticallyOptimalStrategy:

    def test_policy(self, symbols=["JPM"], start_date=dt.datetime(2010, 1, 1), end_date=dt.datetime(2011, 12, 31),
                    start_value=100000):

        dates = pd.date_range(start_date, end_date)
        prices = get_data(symbols, dates, False)
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')

        date_range = pd.date_range(start_date, end_date)
        trades_df = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
        position = 0
        shares_def = 1000
        signal = "BUY"
        symbol = symbols[0]
        length = len(date_range) -1
        for i in range(length):
            trades_df.loc[i, 'Date'] = date_range[i].strftime("%Y-%m-%d")
            trades_df.loc[i, 'Symbol'] = symbol
            if i == 0:
                trades_df.loc[i, 'Order'] = signal
                trades_df.loc[i, 'Shares'] = shares_def
                continue

            if prices.loc[date_range[i].strftime("%Y-%m-%d"), symbol] > prices.loc[
                date_range[i+1].strftime("%Y-%m-%d"), symbol]:
                shares_def = 1000 - position

            elif prices.loc[date_range[i].strftime("%Y-%m-%d"), symbol] < prices.loc[
                date_range[i+1].strftime("%Y-%m-%d"), symbol]:
                shares_def = -1000 - position

            elif prices.loc[date_range[i].strftime("%Y-%m-%d"), symbol] == prices.loc[
                date_range[i+1].strftime("%Y-%m-%d"), symbol]:
                shares_def = -1000 - position

            position = position + shares_def
            if shares_def < 0:
                signal = "SELL"
            else:
                signal = "BUY"
            trades_df.loc[i, 'Order'] = signal
            trades_df.loc[i, 'Shares'] = shares_def

        return trades_df


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mmahalingam6"


def construct_benchmark(start_value, start_date, end_date, symbol, benchmark_shares):
    date_range = pd.date_range(start_date, end_date)
    trades_df = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    # split and invest for each day for one year, until JPM reaches 1000 shares
    benchmark_shares_temp = 0
    shares = 3
    for i in range(len(date_range)):
        benchmark_shares_temp += shares
        if benchmark_shares_temp >= benchmark_shares:
            shares = (benchmark_shares_temp - benchmark_shares) if benchmark_shares_temp - benchmark_shares < 3 else 0
        trades_df.loc[i, 'Date'] = date_range[i].strftime("%Y-%m-%d")
        trades_df.loc[i, 'Symbol'] = symbol
        trades_df.loc[i, 'Order'] = "BUY"
        trades_df.loc[i, 'Shares'] = shares
    # print(f"construct_benchmark : \n {trades_df}")
    portvals_bm_df = compute_portvals(trades_df, start_value, 0.00, 0.00, "Benchmark")
    return portvals_bm_df


def generate_plot(portvals_tos_df, portvals_bm_df):
    plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(211)
    ax1.set_title("Theoretically Optimal Strategy vs Benchmark", fontsize=14)
    ax1.plot(portvals_tos_df, label="TOS", color="red")
    ax1.plot(portvals_bm_df, label="Benchmark", color="purple")
    ax1.set_ylabel('Normalized Portfolio Value', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
    ax1.legend()
    ax1.grid()
    ax1.margins(x=0)
    plt.savefig('mmahalingam6_project6_6.png')
    ax1.remove()
    plt.clf()
    plt.close()


def tos_report():
    start_value = 100000
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 12, 1)
    symbols = ["JPM"]

    # Theoretically Optimal Strategy
    tos = TheoreticallyOptimalStrategy()
    trades_df = tos.test_policy(symbols, start_date, end_date, start_value)
    portvals_tos_df = compute_portvals(trades_df, start_value, 0.00, 0.00, "TOS")
    # print(f"portvals_tos_df \n {portvals_tos_df}")

    # benchmark construction
    benchmark_shares = 1000
    portvals_bm_df = construct_benchmark(start_value, start_date, end_date, symbols[0], benchmark_shares)
    # print(f"portvals_bm_df \n {portvals_bm_df}")

    # normalize value
    portvals_tos_df = portvals_tos_df / portvals_tos_df.iloc[0]
    portvals_bm_df = portvals_bm_df / portvals_bm_df.iloc[0]

    # generate graph
    generate_plot(portvals_tos_df, portvals_bm_df)


if __name__ == "__main__":
    tos_report()
