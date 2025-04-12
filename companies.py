import pandas as pd
import yfinance as yf

# Get S&P 500 tickers from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]  # First table on the page
tickers = sp500_table['Symbol'].tolist()

# Save to a file
with open("sp500_tickers.txt", "w") as f:
    for ticker in tickers:
        f.write(f"{ticker}\n")

print(f"Saved {len(tickers)} tickers to sp500_tickers.txt")