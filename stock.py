import numpy as np
from import_data import import_data
from chart import Chart
import pandas as pd


# Stock Obj
class Stock:

    def __init__(self, ticker, period):

        self.ticker = ticker
        self.period = period
        self.charts = {}
        self.timeframes = ['1d', '1h']

        # Load in charts for each timeframe
        for timeframe in self.timeframes:
            data = import_data(ticker, timeframe, period)
            self.charts[timeframe] = Chart(timeframe, data)

        self.observation_space = self.create_observation_space()

    # Create Chart Obj
    def load_chart(self, ticker, timeframe):
        data = import_data(ticker, timeframe, self.period)
        return Chart(timeframe, data)

    def create_observation_space(self):

        # New observation space to contain hourly and expanded daily spaces (x2 size)
        expanded_space = np.zeros(
            (self.charts['1h'].observation_space.shape[0], self.charts['1h'].observation_space.shape[1] * 2)
        )

        # Add values line by line
        for i in range(expanded_space.shape[0]):

            # Use range between current date and date of next candle
            current_week = self.charts['1h'].dates[i].week
            current_year = self.charts['1h'].dates[i].year

            for n in range(len(self.charts['1d'].dates)):
                if self.charts['1d'].dates[n].week == current_week and self.charts['1d'].dates[n].year == current_year:
                    expanded_space[i, :round(expanded_space.shape[1] / 2)] = self.charts['1d'].observation_space[n, :]
                    expanded_space[i, round(expanded_space.shape[1] / 2):] = self.charts['1h'].observation_space[i, :]
                    break

        return expanded_space