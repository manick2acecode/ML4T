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

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, symbol_to_path
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.lines import Line2D

# Simple Moving Average
def get_indi_sma(process_df, lookback, gen_plot, symbol):
    normalized_process_df = process_df[symbol] / process_df[symbol][0]
    sma = process_df.copy();
    normalized_sma = sma[symbol] / sma[symbol][0]
    normalized_sma = normalized_sma.rolling(window=lookback, min_periods=lookback).mean()
    sma_ratio = normalized_process_df / normalized_sma

    if gen_plot:
        plt.figure(figsize=(14, 8))

        ax1 = plt.subplot(211)
        ax1.set_title("Prices with {} days SMA".format(lookback).format(lookback), fontsize=14)
        ax1.plot(normalized_process_df, label="Normalized Prices", color="orange")
        ax1.plot(normalized_sma, label="{} days SMA".format(lookback), color="blue")
        ax1.set_ylabel('Normalized Price', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax1.legend()
        ax1.grid()
        ax1.margins(x=0)
        plt.subplots_adjust(hspace=.5)

        ax2 = plt.subplot(212)
        ax2.set_title("Prices/SMA Ratio", fontsize=14)
        ax2.plot(sma_ratio, label="Prices/SMA Ratio", color="purple")
        ax2.set_ylabel('Ratio', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.axhline(1.2, color="tab:red", ls="--")
        ax2.axhline(0.8, color="tab:green", ls="--")
        ax2.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax2.legend()
        ax2.grid()
        ax2.margins(x=0)
        plt.savefig('mmahalingam6_project6_1.png')
        ax1.remove()
        ax2.remove()
        plt.clf()
        plt.close()

        pass

    return process_df

#Bollinger Bands
def get_indi_bollinger_band(process_df, lookback, gen_plot, symbol):
    normalized_process_df = process_df[symbol] / process_df[symbol][0]
    sma = process_df.copy();
    normalized_sma = sma[symbol] / sma[symbol][0]
    normalized_sma = normalized_sma.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = normalized_process_df.rolling(window=lookback, min_periods=lookback).std()
    top_band = normalized_sma + (2 * rolling_std)
    bottom_band = normalized_sma - (2 * rolling_std)
    bb_percent = (normalized_process_df - bottom_band) / (top_band - bottom_band)

    if gen_plot:
        plt.figure(figsize=(14, 8))

        ax1 = plt.subplot(211)
        ax1.set_title("Bollinger Bands with lookback of {} days".format(lookback), fontsize=14)
        ax1.plot(normalized_process_df, label="Normalized Prices", color="orange")
        ax1.plot(top_band, label="Top Band", color="red")
        ax1.plot(normalized_sma, label="SMA", color="blue")
        ax1.plot(bottom_band, label="Bottom Band", color="red")
        ax1.set_ylabel('Normalized Price', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax1.legend()
        ax1.grid()
        ax1.margins(x=0)
        plt.subplots_adjust(hspace=.5)

        ax2 = plt.subplot(212)
        ax2.set_title("Bollinger Bands %", fontsize=14)
        ax2.plot(bb_percent, label="BB%", color="blue")
        ax2.set_ylabel('BBP', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.axhline(1.0, color="tab:red", ls="--")
        ax2.axhline(0.0, color="tab:green", ls="--")
        ax2.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax2.legend()
        ax2.grid()
        ax2.margins(x=0)
        plt.savefig('mmahalingam6_project6_2.png')
        ax1.remove()
        ax2.remove()
        plt.clf()
        plt.close()

        pass

    return process_df


def get_indi_macd(process_df, gen_plot, symbol):
    normalized_process_df = process_df[symbol] / process_df[symbol][0]
    ema = process_df.copy();
    normalized_ema = ema[symbol] / ema[symbol][0]
    normalized_ema_12 = normalized_ema.rolling(window=12, min_periods=12).mean()
    normalized_ema_26 = normalized_ema.rolling(window=26, min_periods=26).mean()
    macd = normalized_ema_12 - normalized_ema_26
    signal = macd.rolling(window=9, min_periods=9).mean()

    if gen_plot:
        plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(211)
        ax1.set_title("MACD with EMA between 12 and 26", fontsize=14)
        ax1.plot(normalized_process_df, label="Normalized Prices", color="blue")
        ax1.plot(normalized_ema_12, label="12 days EMA", color="red")
        ax1.plot(normalized_ema_26, label="26 days EMA", color="orange")
        ax1.set_ylabel('Normalized Price', fontsize=12)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax1.legend()
        ax1.grid()
        ax1.margins(x=0)
        plt.subplots_adjust(hspace=.5)

        ax2 = plt.subplot(212)
        ax2.set_title("MACD Signal", fontsize=14)
        ax2.plot(macd, label="MACD", color="orange")
        ax2.plot(signal, label="Signal", color="purple")
        ax2.set_ylabel('Normalized Price', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax2.legend()
        ax2.grid()
        ax2.margins(x=0)
        plt.savefig('mmahalingam6_project6_3.png')
        ax1.remove()
        ax2.remove()
        plt.clf()
        plt.close()

        pass

    return process_df


def get_indi_stochastic_oscillator(process_df, lookback, gen_plot, symbol, dates):
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv(
        symbol_to_path(symbol),
        index_col="Date",
        parse_dates=True,
        usecols=["Date", "Open", "High", "Low", "Close", "Adj Close"],
        na_values=["nan"])
    df = df.join(df_temp)
    prices = df.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices.dropna()

    sto_osc_copy = prices.copy();

    high_roll = sto_osc_copy["High"].rolling(lookback).max()
    low_roll = sto_osc_copy["Low"].rolling(lookback).min()
    # Fast stochastic indicator
    sto_osc_copy["%K"] = ((sto_osc_copy["Close"] - low_roll) / (high_roll - low_roll)) * 100
    # slow stochastic indicator
    sto_osc_copy["%D"] = sto_osc_copy["%K"].rolling(lookback).mean()

    if gen_plot:
        plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(211)
        ax1.set_title("{} Prices".format(symbol), fontsize=14)
        ax1.plot(sto_osc_copy["Adj Close"], label="{} Prices".format(symbol), color="blue")
        ax1.set_ylabel("Prices", fontsize=12)
        ax1.set_xlabel("Date", fontsize=12)
        ax1.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax1.legend(loc="best")
        ax1.grid()
        ax1.margins(x=0)
        plt.subplots_adjust(hspace=.5)

        ax2 = plt.subplot(212)
        ax2.set_title("Stochastic Oscillator ({} days lookback)".format(lookback), fontsize=14)
        ax2.plot(sto_osc_copy["%K"], label="%K", color="orange")
        ax2.plot(sto_osc_copy["%D"], label="%D", color="purple")
        ax2.set_ylabel('Stochastic Oscillator', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.axhline(80, color="tab:red", ls="--")
        ax2.axhline(20, color="tab:green", ls="--")
        ax2.legend()
        custom_lines = [
            Line2D([0], [0], color="tab:orange", lw=4),
            Line2D([0], [0], color="tab:purple", lw=4),
            Line2D([0], [0], color="tab:red", lw=4),
            Line2D([0], [0], color="tab:green", lw=4),
        ]
        ax2.legend(custom_lines, ["%K", "%D", "Overbought", "Oversold"], loc="best")
        ax2.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax2.grid()
        ax2.margins(x=0)
        plt.savefig('mmahalingam6_project6_4.png')
        ax1.remove()
        ax2.remove()
        plt.clf()
        plt.close()

        pass

    return process_df


def get_indi_atr(process_df, lookback, gen_plot, symbol, dates):
    df = pd.DataFrame(index=dates)
    df_temp = pd.read_csv(
        symbol_to_path(symbol),
        index_col="Date",
        parse_dates=True,
        usecols=["Date", "Open", "High", "Low", "Close", "Adj Close"],
        na_values=["nan"])
    df = df.join(df_temp)
    prices = df.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices.dropna()

    sto_osc_copy = prices.copy();

    high_low = sto_osc_copy["High"] - sto_osc_copy["Low"]
    high_close = np.abs(sto_osc_copy["High"] - sto_osc_copy["Close"].shift())
    low_close = np.abs(sto_osc_copy["Low"] - sto_osc_copy["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    sto_osc_copy["ATR"] = true_range.rolling(window=lookback, min_periods=lookback).mean()
    # atr_adj_close = sto_osc_copy["Adj Close"]/sto_osc_copy["Adj Close"][0]

    if gen_plot:

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_title("Average True Range", fontsize=14)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        sto_osc_copy["ATR"].plot(ax=ax)
        prices['Close'].plot(ax=ax, secondary_y=True, alpha=0.3)
        ax.grid()
        ax.xaxis.set_major_formatter(DateFormatter('%b-%Y'))
        ax.margins(x=0)
        plt.savefig('mmahalingam6_project6_5.png')
        ax.legend(loc="best")
        ax.remove()
        plt.clf()
        plt.close()

        pass

    return process_df


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mmahalingam6"


def indicator_report():
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["JPM"]

    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates, False)
    prices = prices_all[symbols]  # only portfolio symbols
    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')
    prices = prices.dropna()

    lookback = 14

    get_indi_sma(prices, lookback, True, symbols[0])

    get_indi_bollinger_band(prices, lookback, True, symbols[0])

    get_indi_macd(prices, True, symbols[0])

    get_indi_stochastic_oscillator(prices, lookback, True, symbols[0], dates)

    get_indi_atr(prices, lookback, True, symbols[0], dates)


if __name__ == "__main__":
    indicator_report()
