import numpy as np


class Chart:

    def __init__(self, timeframe, data):
        self.timeframe = timeframe
        self.data = data
        self.dates = data.index
        self.opens = np.array(data['Open'])
        self.closes = np.array(data['Close'])
        self.highs = np.array(data['High'])
        self.lows = np.array(data['Low'])
        self.volume = np.array(data['Volume'])
        self.candles = self.get_candle_colours()
        self.observation_space = self.observation_space()

    def get_candle_colours(self):
        colours = []
        for i in range(len(self.closes)):
            if i == 0:
                if self.closes[i] > self.opens[i]:
                    colours.append(1)
                else:
                    colours.append(0)
            else:
                if self.closes[i] > self.opens[i]:
                    colours.append(1)
                else:
                    colours.append(0)
        return colours

    def get_mav(self, mav, data):
        values = []
        for i in range(len(data)):
            if i < mav - 1:
                values.append(sum(data[:i + 1]) / (i + 1))
            else:
                values.append(sum(data[i - mav + 1: i + 1]) / mav)
        return values

    def get_rmav(self, mav, norm_size, denom_data):
        mav = np.array(self.get_mav(mav, self.closes))
        rmav = np.array(np.array(denom_data) - mav)
        normed = [0 for _ in range(norm_size)]

        for i in range(norm_size, len(rmav)):
            normed.append((rmav[i] - min(rmav[i-norm_size:i+1])) / (max(rmav[i-norm_size:i+1]) - min(rmav[i-norm_size:i+1])))

        return normed

    def get_dy(self, data):
        dy = []
        for i in range(len(data)):
            if i == 0:
                dy.append(0)
            else:
                dy.append(data[i] - data[i-1])
        dy = np.array(dy)
        dy = (dy - min(dy)) / (max(dy) - min(dy))
        return dy

    def get_cci(self, mav):
        t_price = [np.average([self.highs[i], self.lows[i], self.closes[i]]) for i in range(len(self.closes))]
        ma = self.get_mav(mav, t_price)
        mean_dev = self.get_mav(mav, [np.abs(t_price[i] - ma[i]) for i in range(len(ma))])

        cci = []
        for i in range(len(ma)):
            if i == 0:
                cci.append(0.5)
            else:
                if mean_dev == 0:
                    cci.append(0.5)
                else:
                    val = (t_price[i] - ma[i]) / (0.015 * mean_dev[i])
                    if val < -200:
                        val = -200
                    elif val > 200:
                        val = 200

                    val = (val - (-200)) / (200 - (-200))

                    cci.append(val)

        cci = (np.array(cci) - min(cci)) / (max(cci) - min(cci))
        return cci

    def get_rsi(self, mav):
        rsi = [0 for _ in range(mav)]
        for i in range(mav, len(self.closes)):
            ups = []
            downs = []
            for n in range(i, i-mav, -1):
                U = self.closes[n] - self.closes[n-1] if self.candles[n] == 1 else 0
                D = abs(self.closes[n] - self.closes[n-1]) if self.candles[n] == 0 else 0
                ups.append(U)
                downs.append(D)

            avgU = np.mean(ups)
            avgD = np.mean(downs)

            if avgD == 0:
                rsi.append(50)
            else:
                RS = avgU / avgD
                rsi.append(100 - 100 / (1 + RS))

        # Normalize rsi from 0-100 range
        rsi = (np.array(rsi) - min(rsi)) / (max(rsi) - min(rsi))

        return rsi

    # def vwap(self, period):
    #     vwap = [self.closes[i] for i in range(period)]
    #     for i in range(period, len(self.closes)):
    #         vwap.append((np.mean([self.closes[i], self.highs[i], self.lows[i]]) * self.volume[i]) / np.mean(self.volume[i-period:i]))
    #     return vwap

    def candle_classifier(self):
        pass

    def normed_price(self, data, batch_norm):
        normed_data = []
        for i in range(len(data)):
            if batch_norm > i:

                normed = (data[i] - min(data[:i+1])) / (max(data[:i+1] - min(data[:i+1])))
                normed_data.append(normed)
            else:
                normed = (data[i] - min(data[i-batch_norm:i+1])) / (max(data[i-batch_norm:i+1]) - min(data[i-batch_norm:i+1]))
                normed_data.append(normed)
        return normed_data

    def observation_space(self):

        # closes2 = self.normed_price(self.closes, 2)
        # closes8 = self.normed_price(self.closes, 8)
        # closes16 = self.normed_price(self.closes, 16)
        # closes32 = self.normed_price(self.closes, 32)
        #
        # lows2 = self.normed_price(self.lows, 2)
        # lows8 = self.normed_price(self.lows, 8)
        # lows16 = self.normed_price(self.lows, 16)
        # lows32 = self.normed_price(self.lows, 32)
        #
        # highs2 = self.normed_price(self.highs, 2)
        # highs8 = self.normed_price(self.highs, 8)
        # highs16 = self.normed_price(self.highs, 16)
        # highs32 = self.normed_price(self.highs, 32)
        #
        # rsi1 = self.get_rsi(20)
        # rsi2 = self.get_rsi(50)
        # rsi3 = self.get_rsi(120)
        #
        # cci1 = self.get_cci(20)
        # cci2 = self.get_cci(50)
        # cci3 = self.get_cci(120)

        ohlc = np.array([self.opens, self.highs, self.lows, self.closes])

        past_opens = [(self.opens[i-1] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        past_highs = [(self.highs[i-1] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        past_lows = [(self.lows[i-1] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        past_closes = [(self.closes[i-1] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        opens = [(self.opens[i] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        highs = [(self.highs[i] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        lows = [(self.lows[i] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]
        closes = [(self.closes[i] - np.min(ohlc[:, i-1:i+1])) / (np.max(ohlc[:, i-1:i+1]) - np.min(ohlc[:, i-1:i+1])) if i > 0 else 0 for i in range(len(self.closes))]

        rsi1 = self.get_rsi(20)
        rsi2 = self.get_rsi(50)
        rsi3 = self.get_rsi(120)

        cci1 = self.get_cci(20)
        cci2 = self.get_cci(50)
        cci3 = self.get_cci(120)

        observation_space = np.array([past_opens, past_highs, past_lows, past_closes,
                                      opens, highs, lows, closes,
                                      rsi1, rsi2, rsi3, cci1, cci2, cci3], dtype=np.float64)
        observation_space[np.isnan(observation_space)] = 0

        return observation_space.transpose()