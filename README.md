# 📊 Stock Analyzer 🧠

A Flask web app that analyzes historical stock performance and news sentiment. Users select a company and date range to view metrics like returns, volatility, Sharpe ratio, predictions, and optional sentiment analysis.

## 🚀 Features
- ✅ Searchable dropdown with top 20 popular stocks prioritized
- 📈 Daily return, volatility & Sharpe ratio analysis
- 📉 Price prediction using linear regression
- 💬 Optional news sentiment (via NewsAPI + TextBlob)
- 🌐 Clean Flask-powered web interface

## 🔴 Live Demo
```
https://stock-analyzer-49dn.onrender.com
```

## 📁 Project Structure
```
stock-analyzer/
├── app.py
├── .env
├── requirements.txt
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
├── README.md
```

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-analyzer.git
   cd stock-analyzer
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file**
   ```env
   TWELVE_API_KEY=your_twelvedata_api_key
   NEWS_API_KEY=your_newsapi_key
   ```

5. **Run the app**
   ```bash
   python app.py
   ```

6. **Visit in browser**
   ```
   http://127.0.0.1:5000/
   ```

## 📌 Notes
- The stock list is pulled using the Twelve Data API.
- Sentiment analysis is optional to conserve API quota (100 req/day on NewsAPI).
- Prediction is a simple linear regression (for demo purposes).

## 📜 License
This project is for educational/demo purposes. MIT licensed. Feel free to use and modify it.
