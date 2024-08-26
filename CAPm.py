import yfinance as yf
import numpy as np
import pandas as pd

def get_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df['Close']

def calculate_annual_return(prices):
    daily_returns = prices.pct_change().dropna()
    annual_return = (1 + daily_returns.mean())**252 - 1
    return annual_return

def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = market_returns.var()
    return covariance / market_variance

def capm(stock_return, market_return, beta, risk_free_rate):
    return risk_free_rate + beta * (market_return - risk_free_rate)

def mean_reversion(stock_prices):
    mean_price = stock_prices.mean()
    current_price = stock_prices.iloc[-1]  # Access the last price using .iloc to avoid the warning
    return mean_price, current_price

def main():
    ticker = input("Enter the stock ticker: ").upper()
    market_ticker = '^GSPC'  # S&P 500 index used as the market proxy
    risk_free_rate = 0.01  # Example risk-free rate (1%)

    # Get stock data
    stock_prices = get_stock_data(ticker)
    market_prices = get_stock_data(market_ticker)

    # Calculate annual returns
    stock_return = calculate_annual_return(stock_prices)
    market_return = calculate_annual_return(market_prices)

    # Calculate Beta
    stock_returns = stock_prices.pct_change().dropna()
    market_returns = market_prices.pct_change().dropna()
    beta = calculate_beta(stock_returns, market_returns)

    # CAPM
    expected_return_capm = capm(stock_return, market_return, beta, risk_free_rate)

    # Mean Reversion
    mean_price, current_price = mean_reversion(stock_prices)

    # Print results
    print(f"\nStock Ticker: {ticker}")
    print(f"Annual Stock Return: {stock_return:.2%}")
    print(f"Annual Market Return (S&P 500): {market_return:.2%}")
    print(f"Calculated Beta: {beta:.2f}")
    print(f"Expected Return using CAPM: {expected_return_capm:.2%}")

    print(f"\nMean Reversion Analysis:")
    print(f"Mean Price: {mean_price:.2f}")
    print(f"Current Price: {current_price:.2f}")
    if current_price > mean_price:
        print(f"{ticker} is potentially overvalued based on mean reversion.")
    else:
        print(f"{ticker} is potentially undervalued based on mean reversion.")

    # Cross-reference conclusion
    overvalued = sum([
        current_price > expected_return_capm,
        current_price > mean_price,
    ])

    if overvalued == 2:
        print(f"\nFinal Conclusion: {ticker} appears to be overvalued based on the models.")
    elif overvalued == 0:
        print(f"\nFinal Conclusion: {ticker} appears to be undervalued based on the models.")
    else:
        print(f"\nFinal Conclusion: {ticker} is fairly valued based on the models.")

if __name__ == "__main__":
    main()
