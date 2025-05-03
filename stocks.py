import yfinance as yf

try:
    data = yf.download(["AAPL"], start="2020-01-01", end="2025-04-29")
    data.to_csv("apply_stock.csv")
    print("Your file is ready!")
except Exception as e:
    print(f"An error occurred: {e}")
