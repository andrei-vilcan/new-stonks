import pandas as pd
import yfinance as yf


def import_data(ticker, interval, period):
    if not period:
        period = '1000d'

    if interval == '4h':
        try:
            df = yf.download(tickers=ticker, interval='1h', period=period)
            new_df = pd.DataFrame(columns=df.columns)
            for i in range(len(df)):
                if not i % 4:
                    new_df = new_df.append(df.iloc[i])
            return new_df
        except:
            print(f'Could not load data for: {ticker}.')
    elif interval == '8h':
        try:
            df = yf.download(tickers=ticker, interval='1h', period=period)
            new_df = pd.DataFrame(columns=df.columns)
            for i in range(len(df)):
                if not i % 8:
                    new_df = new_df.append(df.iloc[i])
            return new_df
        except:
            print(f'Could not load data for: {ticker}.')
    else:
        try:
            df = yf.download(tickers=ticker, interval=interval, period=period)
            # stock = yf.Ticker(ticker=ticker)
            # stock.history()
            return df
        except:
            print(f'Could not load data for: {ticker}.')