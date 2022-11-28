from stock import Stock
from gym import Env
import matplotlib.pyplot as plt
import numpy as np
import random


# Trading Environment Convolution Model
class TradingEnv(Env):
    def __init__(self, timeframe, period, ticks, batch_size=64, cash=1000, tx_fee=0.0075, min_stock_size=1000,
                 trading_period=300, hold_decay=0.0005):
        # Load in stock data
        self.tickers = ticks
        self.timeframe = timeframe
        self.period = period
        self.min_stock_size = min_stock_size
        self.trading_period = trading_period
        self.hold_decay = hold_decay

        self.stocks = []
        self.load_stocks()

        # Variables for environment
        self.starting_cash = cash
        self.tx_fee = tx_fee
        self.batch_size = batch_size

        self.action_space = [0, 1, 2] # 0 short, 1 long, 2 none
        self.action_space_size = np.shape(self.action_space)
        self.reset()

    def reset(self):
        # Creates custom observation space for every reset using a random chart
        self.choose_stock()

        # Other variables to reset
        self.state = self.observation_space[self.epoch-self.batch_size:self.epoch]
        self.cash = self.starting_cash
        self.actions = []
        self.reward = 0
        self.prev_reward = 1.0
        self.info = {}
        self.done = 0

        return self.state

    def step(self, action):
        # Calculate reward (% gain) between current state and next
        p_change = (self.closes[self.epoch+1] - self.closes[self.epoch]) / self.closes[self.epoch]
        # p_change = np.divide(np.subtract(self.closes[i+1], self.closes[i]), self.closes[i])

        self.current_state = self.observation_space[self.epoch - self.batch_size:self.epoch]

        # Calculate the reward based on next days stock data and action taken, the first set of reward calclations
        # for each outcome is based on an actual trading system (% changes), the second set of reward calculations are
        # based on simply adding consecutive percentages (not how % gains/trading works, but agent yielded better
        # results compared to % gain method)
        if action == 0:
            if self.actions:
                if self.actions[-1] == 0:
                    # ind = -1
                    # count = 0
                    # while self.actions[ind] == 0:
                    #     count += 1
                    #     if -ind == len(self.actions):
                    #         break
                    #     ind -= 1
                    self.reward = (-p_change) # - np.log10(count) / 2_000
                    # self.reward = self.prev_reward * (1 + (-p_change)) * (1 - self.hold_decay)
                # If different position as previous time step, apply trading fee
                else:
                    self.reward = (-p_change) - self.tx_fee
                    # self.reward = self.prev_reward * (1 - self.tx_fee) * (1 + (-p_change))
            else:
                self.reward = (-p_change) - self.tx_fee
                # self.reward = (1 + (-p_change)) * (1 - self.tx_fee)
        elif action == 1:
            if self.actions:
                if self.actions[-1] == 1:
                    # ind = -1
                    # count = 0
                    # while self.actions[ind] == 1:
                    #     count += 1
                    #     if -ind == len(self.actions):
                    #         break
                    #     ind -= 1
                    self.reward = p_change # - np.log10(count) / 2_000
                    # self.reward = self.prev_reward * (1 + p_change) * (1 - self.hold_decay)
                # If different position as previous time step, apply trading fee
                else:
                    self.reward = p_change - self.tx_fee
                    # self.reward = self.prev_reward * (1 - self.tx_fee) * (1 + p_change)
            else:
                self.reward = p_change - self.tx_fee
                # self.reward = (1 + p_change) * (1 - self.tx_fee)
        elif action == 2:
            if self.actions:
                if self.actions[-1] == 2:
                    # ind = -1
                    # count = 0
                    # while self.actions[ind] == 2:
                    #     count += 1
                    #     if -ind == len(self.actions):
                    #         break
                    #     ind -= 1
                    self.reward = 0 # - np.log10(count) / 1_000
                    # self.reward = self.prev_reward * (1 - self.hold_decay)
                # If different position as previous time step, apply trading fee
                else:
                    self.reward = 0 - self.tx_fee
                    # self.reward = self.prev_reward * (1 - self.tx_fee)
            else:
                self.reward = 0
                # self.reward = 1
        # if action == 0:
        #     if self.actions:
        #         if self.actions[-1] == 0:
        #             self.reward = 0
        #         elif self.actions[-1] == 1:
        #             self.reward = (self.opens[self.epoch+1] - self.entry) / self.entry - self.tx_fee
        #             self.entry = self.opens[self.epoch+1]
        #         elif self.actions[-1] == 2:
        #             self.reward = -self.tx_fee
        #             self.entry = self.opens[self.epoch+1]
        #     else:
        #         self.reward = -self.tx_fee
        #         self.entry = self.opens[self.epoch+1]
        # elif action == 1:
        #     if self.actions:
        #         if self.actions[-1] == 0:
        #             self.reward = -(self.opens[self.epoch+1] - self.entry) / self.entry - self.tx_fee
        #             self.entry = self.opens[self.epoch+1]
        #         elif self.actions[-1] == 1:
        #             self.reward = 0
        #         elif self.actions[-1] == 2:
        #             self.reward = -self.tx_fee
        #             self.entry = self.opens[self.epoch+1]
        #     else:
        #         self.reward = -self.tx_fee
        #         self.entry = self.opens[self.epoch + 1]
        # elif action == 2:
        #     if self.actions:
        #         if self.actions[-1] == 0:
        #             self.reward = -(self.opens[self.epoch+1] - self.entry) / self.entry - self.tx_fee
        #             self.entry = self.opens[self.epoch+1]
        #         elif self.actions[-1] == 1:
        #             self.reward = (self.opens[self.epoch+1] - self.entry) / self.entry - self.tx_fee
        #             self.entry = self.opens[self.epoch + 1]
        #         elif self.actions[-1] == 2:
        #             self.reward = 0
        #     else:
        #         self.reward = 0
        else:
            raise SystemExit('Action should be 0 (short), 1 (long), 2 (no position).')

        # Add action to previous action
        # self.observation_space[self.epoch, -1] = action

        self.epoch += 1
        self.next_state = self.observation_space[self.epoch-self.batch_size:self.epoch]
        self.actions.append(action)

        # # Doubles the punishment if a wrong choice is made
        # if self.reward < self.prev_reward:
        #     self.reward = self.reward - (self.prev_reward - self.reward)

        if self.epoch == self.start_ind + self.trading_period - 1:
            self.done = 1

        self.reward = round(self.reward, 5)
        self.prev_reward = self.reward
        self.info = {'State': self.current_state, 'Reward': self.reward, 'Done': self.done}

        return self.next_state, self.reward, self.done, self.info

    def render(self, **kwargs):
        # Display custom candlestick chart
        longs = []
        shorts = []
        out = []
        for i in range(len(self.actions)):
            if i == 0:
                if self.actions[i] == 0:
                    shorts.append(i)
                elif self.actions[i] == 1:
                    longs.append(i)
                else:
                    out.append(i)
            elif self.actions[i] == 0:
                if self.actions[i-1] != 0:
                    shorts.append(i)
            elif self.actions[i] == 1:
                if self.actions[i-1] != 1:
                    longs.append(i)
            elif self.actions[i] == 2:
                if self.actions[i-1] != 2:
                    out.append(i)
        plt.title(self.stock.ticker + ': ' + str(self.start_ind) + '-' + str(self.start_ind + self.trading_period))
        plt.plot(list(self.closes[self.start_ind:self.epoch]), c='m')
        for long in longs:
            plt.axvline(long, c='g')
        for short in shorts:
            plt.axvline(short, c='r')
        for o in out:
            plt.axvline(o, c='k')
        plt.pause(0.0001)
        plt.close()

    def load_stocks(self):
        chart_count = 0

        for ticker in self.tickers:
            stock = Stock(ticker, self.period)
            if len(stock.observation_space) > self.min_stock_size:
                # vol = True
                # num_replaced = 0
                # for i in range(len(stock.charts[self.timeframe].volume)):
                #     if i > 0:
                #         if stock.charts[self.timeframe].volume[i] == 0:
                #             stock.charts[self.timeframe].volume[i] = stock.charts[self.timeframe].volume[i-1]
                #             num_replaced += 1
                #     elif i == 0:
                #         if stock.charts[self.timeframe].volume[i] == 0:
                #             stock.charts[self.timeframe].volume[i] = stock.charts[self.timeframe].volume[i+1]
                #             num_replaced += 1

                # if vol and num_replaced < 20:
                self.stocks.append(stock)
                chart_count += 1

        print(f'Training model with {len(self.stocks)} stock(s).')

    def choose_stock(self):
        # Load in random stock data
        self.stock = random.sample(self.stocks, 1)[0]
        self.chart = self.stock.charts[self.timeframe]

        self.opens = self.chart.opens
        self.highs = self.chart.highs
        self.lows = self.chart.lows
        self.closes = self.chart.closes

        self.start_ind = 199 + self.trading_period * random.sample(range((len(self.closes) - 200) // self.trading_period), 1)[0]
        self.epoch = self.start_ind

        # Update observation space
        self.observation_space = self.chart.observation_space
        self.observation_space_size = self.observation_space.shape

        self.previous_action = 0


## For FC Model ##
# class TradingEnv(Env):
#     def __init__(self, timeframe, period, ticks, batch_size=64, cash=1000, tx_fee=0.0075, min_stock_size=1000,
#                  trading_period=300):
#         # Load in stock data
#         self.tickers = ticks
#         self.timeframe = timeframe
#         self.period = period
#         self.min_stock_size = min_stock_size
#         self.trading_period = trading_period
#
#         self.stocks = []
#         self.load_stocks()
#
#         # Variables for environment
#         self.starting_cash = cash
#         self.tx_fee = tx_fee
#         self.batch_size = batch_size
#
#         self.action_space = [0, 1, 2]
#         self.action_space_size = np.shape(self.action_space)
#         # Observation space for batched data
#         # self.observation_space = np.array([self.chart.observation_space()[i:i+self.batch_size] for i in
#         #                                    range(0, self.trading_length - self.batch_size)])
#
#         self.reset()
#
#     def reset(self):
#         # Creates custom observation space for every reset using a random chart
#         self.choose_stock()
#
#         # Other variables to reset
#         self.state = self.observation_space[self.epoch]
#         self.cash = self.starting_cash
#         self.actions = []
#         self.reward = 0
#         self.prev_reward = 0
#         self.info = {}
#         self.done = 0
#
#         return self.state
#
#     def step(self, action):
#         i = self.epoch
#
#         # Calculate reward (% gain) between current state and next
#         p_change = (self.closes[i+1] - self.closes[i]) / self.closes[i]
#         # p_change = np.divide(np.subtract(self.closes[i+1], self.closes[i]), self.closes[i])
#
#         # Calculate the reward based on next days stock data and action taken, the first set of reward calclations
#         # for each outcome is based on an actual trading system (% changes), the second set of reward calculations are
#         # based on simply adding consecutive percentages (not how % gains/trading works, but agent yielded better
#         # results compared to % gain method)
#         if action == 0:
#             if self.actions:
#                 if self.actions[-1] == 0:
#                     # self.reward = self.prev_reward * (1 + (-p_change))
#                     self.reward = self.prev_reward + (-p_change)
#                 # If different position as previous time step, apply trading fee
#                 else:
#                     # self.reward = (self.prev_reward * (1 - self.tx_fee)) * (1 + (-p_change))
#                     self.reward = self.prev_reward - self.tx_fee + (-p_change)
#             else:
#                 # self.reward = 1 * (1 + (-p_change)) * (1 - self.tx_fee)
#                 self.reward = 1 + (-p_change) - self.tx_fee # First reward, first action
#         elif action == 1:
#             if self.actions:
#                 if self.actions[-1] == 1:
#                     # self.reward = self.prev_reward * (1 + p_change)
#                     self.reward = self.prev_reward + p_change
#                 # If different position as previous time step, apply trading fee
#                 else:
#                     # self.reward = (self.prev_reward * (1 - self.tx_fee)) * (1 + p_change)
#                     self.reward = self.prev_reward - self.tx_fee + p_change
#             else:
#                 # self.reward = 1 * (1 + (p_change)) * (1 - self.tx_fee)
#                 self.reward = 1 + p_change - self.tx_fee # First reward, first action
#         elif action == 2:
#             if self.actions:
#                 if self.actions[-1] == 2:
#                     self.reward = self.prev_reward - 0.0005
#                 # If different position as previous time step, apply trading fee
#                 else:
#                     # self.reward = self.prev_reward * (1 - self.tx_fee)
#                     self.reward = self.prev_reward - self.tx_fee
#             else:
#                 self.reward = 1 # First reward, first action
#         else:
#             raise SystemExit('Action should be either 1 (long) or 0 (short).')
#
#         # self.cmltv_score += reward
#
#         self.epoch += 1
#         self.actions.append(action)
#
#         # -1 due to epoch += 1 above
#         self.current_state = self.observation_space[self.epoch-1]
#
#         # Update value previous action in observation space
#         # self.observation_space[self.epoch][-1] = self.actions[-1] # This previous action system is definitely a huge
#         # source of errors, check observation space in stock.py as well
#
#         self.next_state = self.observation_space[self.epoch]
#
#         if self.epoch == self.start_ind + self.trading_period - 1:
#             self.done = 1
#
#         self.prev_reward = self.reward
#
#         self.info = {'State': self.current_state, 'Reward': self.reward, 'Done': self.done}
#
#         return self.next_state, self.reward, self.done, self.info
#
#     def render(self):
#         # Display custom candlestick chart
#         c = []
#         for a in self.actions:
#             if a == 0:
#                 c.append('r')
#             elif a == 1:
#                 c.append('g')
#             elif a == 2:
#                 c.append('k')
#
#         plt.scatter(x=range(len(self.closes[self.start_ind:self.epoch])),
#                     y=self.closes[self.start_ind:self.epoch], c=c)
#         plt.pause(0.0001)
#         plt.close()
#
#     def load_stocks(self):
#         chart_count = 0
#
#         for ticker in self.tickers:
#             stock = Stock(ticker, self.timeframe, self.period)
#             if len(stock.charts[self.timeframe].closes) > self.min_stock_size:
#                 # if sum([v != 0 for v in stock.charts[self.timeframe].volume]) == len(stock.charts[self.timeframe].volume):
#                 self.stocks.append(stock)
#                 chart_count += 1
#
#         print(f'Training model with {len(self.stocks)} stock(s).')
#
#     def choose_stock(self):
#         # Load in random stock data
#         self.stock = random.sample(self.stocks, 1)[0]
#         self.chart = self.stock.charts[self.timeframe]
#
#         self.opens = self.chart.opens
#         self.highs = self.chart.highs
#         self.lows = self.chart.lows
#         self.closes = self.chart.closes
#
#         self.start_ind = 120 + self.trading_period * random.sample(range((len(self.closes) - 120) // self.trading_period), 1)[0]
#         self.epoch = self.start_ind
#
#         # Update observation space
#         self.observation_space = self.chart.observation_space()
#         self.observation_space_size = self.observation_space.shape
#
#         self.previous_action = 0
