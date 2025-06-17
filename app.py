from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from dotenv import load_dotenv
import os
import plotly.graph_objs as go  

app = Flask(__name__)
load_dotenv()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
sentiment_cache = {}

# Popular + Fallback
def get_symbols():
    popular = [
        ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"), ("GOOGL", "Alphabet Inc."),
        ("AMZN", "Amazon.com Inc."), ("TSLA", "Tesla Inc."), ("META", "Meta Platforms Inc."),
        ("NFLX", "Netflix Inc."), ("NVDA", "NVIDIA Corporation"), ("JPM", "JPMorgan Chase & Co."),
        ("V", "Visa Inc."), ("MA", "Mastercard Inc."), ("DIS", "Walt Disney Co."),
        ("INTC", "Intel Corporation"), ("ADBE", "Adobe Inc."), ("PYPL", "PayPal Holdings Inc."),
        ("PEP", "PepsiCo Inc."), ("KO", "Coca-Cola Co."), ("NKE", "Nike Inc."),
        ("CRM", "Salesforce Inc."), ("CSCO", "Cisco Systems Inc.")
    ]
    all_symbols = []
    for ex in ["NASDAQ", "NYSE"]:
        try:
            r = requests.get(f"https://api.twelvedata.com/stocks?exchange={ex}&apikey={TWELVE_API_KEY}")
            if r.status_code == 200:
                all_symbols += [(d["symbol"], d["name"]) for d in r.json().get("data", []) if "symbol" in d]
        except:
            continue
    existing = set(s[0] for s in popular)
    return popular + [s for s in sorted(set(all_symbols)) if s[0] not in existing]

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
    trace = go.Scatter(
        x=df.index,
        y=df['Adj Close'],
        mode='lines+markers',
        name='Adj Close',
        line=dict(color='navy')
    )

    layout = go.Layout(
        title=f'{symbol} Price Chart',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price'),
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white'
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig.to_html(full_html=False)

def assess_risk(avg_return, volatility, sentiment_score):
    score = 0
    if sentiment_score is None: score += 2
    elif sentiment_score > 0.2: score += 0
    elif sentiment_score < -0.2: score += 2
    else: score += 1
    if volatility > 0.03: score += 2
    elif volatility > 0.015: score += 1
    if avg_return < 0: score += 2
    elif avg_return < 0.0005: score += 1
    if score <= 1: return (1, "🟢 Strong Buy – Very Low Risk")
    elif score == 2: return (2, "🟢 Buy – Low Risk")
    elif score == 3: return (3, "🟡 Hold – Moderate Risk")
    elif score == 4: return (4, "🟠 Avoid – High Risk")
    else: return (5, "🔴 Strong Avoid – Very High Risk")

def get_sentiment(symbol):
    if symbol in sentiment_cache:
        return sentiment_cache[symbol]
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&language=en&pageSize=5"
        r = requests.get(url)
        articles = r.json().get("articles", [])[:5]
        sentiments = [TextBlob(f"{a.get('title', '')} {a.get('description', '')}").sentiment.polarity for a in articles]
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
    current_price, diff, pct = get_current_price(symbol)

    if df.empty or len(df) < 2:
        return render_template("result.html", symbol=symbol, name=company_name,
                               error=f"No price data available for {symbol} in selected date range.",
                               current_price=current_price, diff=diff, pct=pct)

    avg_return = df['Return'].mean()
    volatility = df['Return'].std()
    sharpe = avg_return / volatility if volatility != 0 else 0
    price_chart = plot_prices(df, symbol)
    sentiment = get_sentiment(symbol) if include_sentiment else None
    risk_level, risk_label = assess_risk(avg_return, volatility, sentiment['score'] if sentiment else None)

    return render_template("result.html", symbol=symbol, name=company_name,
                           avg_return=avg_return, volatility=volatility, sharpe=sharpe,
                           risk_level=risk_level, risk_label=risk_label,
                           chart=price_chart, sentiment=sentiment,
                           current_price=current_price, diff=diff, pct=pct)

if __name__ == "__main__":
    app.run(debug=True)
