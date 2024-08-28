import numpy as np
import scipy.stats as si
import yfinance as yf

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes option price for a call or put option.

    Parameters:
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to expiration (in years)
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    option_type : str : "call" for call option, "put" for put option

    Returns:
    float : Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        option_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == "put":
        option_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='1d')
    return df['Close'][0]

def main():
    ticker = input("Enter the stock ticker: ").upper()
    S = get_stock_data(ticker)  # Current stock price
    K = float(input("Enter the option strike price: "))  # Option strike price
    T = float(input("Enter time to expiration in days: ")) / 365  # Convert days to years
    r = 0.01  # Example risk-free interest rate (1%)
    sigma = float(input("Enter the stock volatility (as a decimal): "))  # Volatility (e.g., 0.2 for 20%)
    option_type = input("Enter the option type (call/put): ").lower()  # Option type

    # Calculate option price using Black-Scholes model
    option_price = black_scholes(S, K, T, r, sigma, option_type)

    print(f"\nThe {option_type} option price for {ticker} is: ${option_price:.2f}")

if __name__ == "__main__":
    main()
