from flask import Flask, render_template, request
import yfinance as yf
import matplotlib.pyplot as plt
import io, base64, os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from textblob import TextBlob
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
sentiment_cache = {}


def get_symbols():
    popular = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("GOOGL", "Alphabet Inc."),
        ("AMZN", "Amazon.com Inc."),
        ("TSLA", "Tesla Inc."),
        ("META", "Meta Platforms Inc."),
        ("NFLX", "Netflix Inc."),
        ("NVDA", "NVIDIA Corporation"),
        ("JPM", "JPMorgan Chase & Co."),
        ("V", "Visa Inc."),
        ("MA", "Mastercard Inc."),
        ("DIS", "Walt Disney Co."),
        ("INTC", "Intel Corporation"),
        ("ADBE", "Adobe Inc."),
        ("PYPL", "PayPal Holdings Inc."),
        ("PEP", "PepsiCo Inc."),
        ("KO", "Coca-Cola Co."),
        ("NKE", "Nike Inc."),
        ("CRM", "Salesforce Inc."),
        ("CSCO", "Cisco Systems Inc.")
    ]
    all_symbols = []
    exchanges = ["NASDAQ", "NYSE"]
    for exchange in exchanges:
        url = f"https://api.twelvedata.com/stocks?exchange={exchange}&apikey={TWELVE_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json().get("data", [])
                all_symbols += [(item["symbol"], item["name"]) for item in data if "symbol" in item and "name" in item]
        except:
            continue
    popular_set = set(sym[0] for sym in popular)
    filtered = [s for s in sorted(set(all_symbols)) if s[0] not in popular_set]
    return popular + filtered

    

def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        else:
            return pd.DataFrame()
    df['Return'] = df['Adj Close'].pct_change()
    df.dropna(inplace=True)
    return df

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="2d")
    if len(data) >= 2:
        current = data['Close'].iloc[-1]
        previous = data['Close'].iloc[-2]
        diff = current - previous
        pct_change = (diff / previous) * 100
        return round(current, 2), round(diff, 2), round(pct_change, 2)
    return None, None, None

def plot_prices(df, symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Adj Close'], label='Adj Close', color='navy')
    plt.title(symbol + " Price Chart")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return img

def predict_price(df):
    df['Days'] = np.arange(len(df)).reshape(-1, 1)
    X = df['Days'].values.reshape(-1, 1)
    y = df['Adj Close'].values
    model = LinearRegression().fit(X, y)
    future_day = np.array([[len(df)]])
    pred = model.predict(future_day)
    return float(pred[0])

def get_sentiment(symbol):
    if symbol in sentiment_cache:
        return sentiment_cache[symbol]
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])[:5]
        sentiments = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        avg = sum(sentiments) / len(sentiments) if sentiments else 0
        summary = "🟢 Positive" if avg > 0.1 else "🔴 Negative" if avg < -0.1 else "⚪ Neutral"
        result = {"summary": summary, "score": round(avg, 2)}
        sentiment_cache[symbol] = result
        return result
    except:
        return {"summary": "Sentiment not available", "score": None}

@app.route("/", methods=["GET"])
def index():
    symbols = get_symbols()
    return render_template("index.html", symbols=symbols)

@app.route("/result", methods=["POST"])
def result():
    symbol = request.form["symbol"]
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    include_sentiment = request.form.get("include_sentiment") == "on"

    company_name = dict(get_symbols()).get(symbol, symbol)
    df = fetch_data(symbol, start_date, end_date)

    # Always fetch current price (even if data is missing)
    current_price, diff, pct = get_current_price(symbol)

    if df.empty or len(df) < 2:
        error_msg = f"No price data available for {symbol} in selected date range."
        return render_template("result.html", symbol=symbol,
                               name=company_name,
                               error=error_msg,
                               current_price=current_price,
                               diff=diff,
                               pct=pct)

    avg_return = df['Return'].mean()
    volatility = df['Return'].std()
    sharpe = avg_return / volatility if volatility != 0 else 0
    prediction = predict_price(df)
    price_chart = plot_prices(df, symbol)
    sentiment = get_sentiment(symbol) if include_sentiment else None

    return render_template("result.html", symbol=symbol, name=company_name,
                           avg_return=avg_return,
                           volatility=volatility,
                           sharpe=sharpe,
                           prediction=prediction,
                           chart=price_chart,
                           sentiment=sentiment,
                           current_price=current_price,
                           diff=diff,
                           pct=pct)


if __name__ == "__main__":
    app.run(debug=True)
