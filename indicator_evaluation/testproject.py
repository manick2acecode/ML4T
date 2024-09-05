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

import pandas as pd
import datetime as dt
from util import get_data
import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals
from indicators import indicator_report


def test_code():
    start_value = 100000
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 12, 1)
    symbols = ["JPM"]

    tos_obj = tos.TheoreticallyOptimalStrategy()
    df_trades = tos_obj.test_policy(symbols=symbols, start_date=start_date, end_date=end_date, start_value=start_value)
    portvals_tos_df = compute_portvals(df_trades, 100000, 0.00, 0.00, "TOS")

    # benchmark construction
    benchmark_shares = 1000
    portvals_bm_df = tos.construct_benchmark(start_value, start_date, end_date, symbols[0], benchmark_shares)
    # print(f"portvals_bm_df \n {portvals_bm_df}")

    # normalize value
    portvals_tos_df = portvals_tos_df / portvals_tos_df.iloc[0]
    portvals_bm_df = portvals_bm_df / portvals_bm_df.iloc[0]

    # generate graph
    tos.generate_plot(portvals_tos_df, portvals_bm_df)

    indicator_report()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mmahalingam6"


if __name__ == "__main__":
    test_code()
