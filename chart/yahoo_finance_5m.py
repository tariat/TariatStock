import yfinance as yf

ticker = "005930.KS"  # 삼성전자 (코스피: .KS)
stock = yf.Ticker(ticker)
df = stock.history(period="1d", interval="5m")
print(df)