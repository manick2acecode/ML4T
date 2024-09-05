""""""
"""ML4T-P6: Indicator Simulation.  		  	   		  		 			  		 			     			  	 

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


def compute_portvals(
        order_trades_df,
        start_val=1000000,
        commission=0.00,
        impact=0.00,
        run_type=""
):
    """
    Computes the portfolio values.

    :param run_type: Benchmark or TOS
    :param order_trades_df: constructed as an order file
    :type order_trades_df: dataframe object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # get start date and end date from orders file
    start_date = order_trades_df.loc[0, 'Date']
    end_date = order_trades_df.loc[len(order_trades_df) - 1, 'Date']

    # extract symbols from order file
    extract_symbols = get_symbol_from_file(order_trades_df)
    # print(f"extract_symbols: {extract_symbols}")

    # get price data for the symbols, drop NaN values, set Cash to 1
    sym_prices_df = get_data(extract_symbols, pd.date_range(start_date, end_date), True)
    sym_prices_df = sym_prices_df.copy()
    sym_prices_df['Cash'] = 1.0000
    # print(f"sym_prices_df: \n {len(sym_prices_df)}")
    temp_start_val = start_val

    # create trading dataframe and make it empty
    trading_df = sym_prices_df.copy()
    trading_df.iloc[:, :] = 0

    pd.set_option('display.float_format', '{:.8f}'.format)

    # iterate trading_df to update matrix of transaction cost with or without commission, impact
    for index in trading_df.index:
        for i in range(len(order_trades_df)):
            date = order_trades_df.loc[i, 'Date']
            if pd.to_datetime(index) == pd.to_datetime(date):
                symbol = order_trades_df.loc[i, 'Symbol']
                shares = order_trades_df.loc[i, 'Shares']
                order = order_trades_df.loc[i, 'Order']
                # print(f"symbol: {symbol}, shares: {shares}, order: {order}, price: {sym_prices_df.loc[idx][symbol]}")
                # per transaction cost, symbol price * no. of shares
                per_trans_cost = sym_prices_df.loc[index][symbol] * shares
                # comm_cost = commission + (symbol price * no. of shares * impact)
                per_tranz_comm_cost = commission + (impact * per_trans_cost)
                # print(f"per_tranz_comm_cost : {per_tranz_comm_cost}, per_trans_cost: {per_trans_cost}")
                if order == 'BUY':
                    trading_df.loc[index, symbol] += shares
                    trading_df.loc[index, 'Cash'] = trading_df.loc[index, 'Cash'] - per_trans_cost - per_tranz_comm_cost
                    # print(f"BUY tran_pri : {trading_df.loc[index][symbol]}, cash : {trading_df.loc[index]['Cash']}")
                elif order == 'SELL':
                    trading_df.loc[index, symbol] -= shares
                    trading_df.loc[index, 'Cash'] = trading_df.loc[index, 'Cash'] + per_trans_cost - per_tranz_comm_cost
                    # print(f"SELL tran_pri : {trading_df.loc[index][symbol]}, cash : {trading_df.loc[index]['Cash']}")
                else:
                    print("Data is missing in the file")
    # print(f"trading_df {trading_df}")

    # holdings data frame calculation
    holdings_df = trading_df.copy()
    holdings_df.iloc[:, :] = 0
    holdings_df['Cash'] = temp_start_val
    holdings_df.iloc[0, :] += trading_df.iloc[0, :]
    for i in range(len(holdings_df)):
        if i == 0:
            continue
        holdings_df.iloc[i, :] = holdings_df.iloc[i - 1, :] + trading_df.iloc[i, :]
    # print(f"holdings_df {holdings_df}")

    # values data frame calculation, values = holdings * prices
    values_df = holdings_df * sym_prices_df
    port_val = pd.DataFrame(index=holdings_df.index)
    port_val['Value'] = values_df.sum(axis=1)
    # print(f"port_val \n {port_val}")

    # Metrics calculation starts here
    if isinstance(port_val, pd.DataFrame):
        port_val = port_val[port_val.columns[0]]
    else:
        "warning, code did not return a DataFrame"

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_metrics(port_val.to_frame(), start_val)

    # Print Metrics
    print(f"{run_type} Date Range: {start_date} to {end_date}")
    #print(f"{run_type} Sharpe Ratio of Fund: {sharpe_ratio[0]}")
    print(f"{run_type} Cumulative Return of Fund: {cum_ret}")
    print(f"{run_type} Standard Deviation of Daily Return of Fund: {std_daily_ret[0]}")
    print(f"{run_type} Mean of Daily Return of Fund: {avg_daily_ret[0]}")
    print(f"{run_type} Final Portfolio Value: {round(port_val[-1], 4)}")
    print()
    print()

    return port_val


def get_portfolio_metrics(port_val, start_val):
    # compute daily portfolio values
    port_val_daily = compute_daily_port_values(port_val, start_val)
    # Daily Returns
    port_val_df = compute_daily_returns(port_val)

    # Average Daily Return
    avg_daily_rtn = port_val_df.mean()
    # print(f"avg_daily_rtn: {avg_daily_rtn}")

    # Standard Deviation of Daily Return
    std_daily_rtn = port_val_df.std()
    # print(f"std_daily_rtn: {std_daily_rtn}")

    # Cummulative Return
    cum_return = (port_val_daily[-1] / port_val_daily[0]) - 1
    # print(f"cum_return: {cum_return}")

    # Sharpe Ratio, Risk Adjusted Return, 0%
    # 252 trading day samples
    sharpe_ratio = (np.sqrt(252.0) * (avg_daily_rtn - 0)) / std_daily_rtn
    # print(f"sharpe_ratio: {sharpe_ratio}")

    return cum_return, avg_daily_rtn, std_daily_rtn, sharpe_ratio


def get_symbol_from_file(orders_df):
    symbols = []
    for index in range(len(orders_df)):
        if orders_df.loc[index, 'Symbol'] not in symbols:
            symbols.append(orders_df.loc[index, 'Symbol'])
    return symbols


def compute_daily_port_values(prices, start_val):
    normed_prices = prices / prices.iloc[0]
    pos_val = normed_prices * start_val
    port_val = pos_val.sum(axis=1)
    return port_val


def compute_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0] = 0
    daily_returns = daily_returns[1:]
    return daily_returns


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mmahalingam6"


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    sv = 1000000
    # Process orders
    # portvals = compute_portvals(orders_file="./additional_orders/orders2.csv", start_val=sv)
    # portvals = compute_portvals(orders_file="./additional_orders/orders.csv", start_val=sv)
    # portvals = compute_portvals(orders_file="./additional_orders/orders-short.csv", start_val=sv)
    # portvals = compute_portvals(orders_file="./orders/orders-01.csv", start_val=sv, commission=0.0, impact=0.0)
    portvals = compute_portvals(orders_file="./orders/orders-02.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-03.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-04.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-05.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-06.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-07.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-08.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-09.csv", start_val=sv, commission=0.0, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-10.csv", start_val=sv, commission=9.95, impact=0.005)
    # portvals = compute_portvals(orders_file="./orders/orders-11.csv", start_val=sv, commission=9.95, impact=0.0)
    # portvals = compute_portvals(orders_file="./orders/orders-12.csv", start_val=sv, commission=0.0, impact=0.005)


if __name__ == "__main__":
    test_code()
