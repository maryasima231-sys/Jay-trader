import logging
import numpy as np
import pandas as pd
import requests
import ta
import joblib
import os
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler

TOKEN ="8528363878:AAF29_Ay2PX9A8yldDg8JzDA_4lG2YiAgiE"
TWELVE_DATA_API_KEY ="44d3f7079c8443ce84b59962617e04c9"

PAIRS = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCAD": "USD/CAD"
}

TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h"
}

MODEL_FILE = "ml_model.pkl"
SCALER_FILE = "scaler.pkl"

logging.basicConfig(level=logging.INFO)

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
else:
    model = SGDClassifier(loss="log_loss")
    scaler = StandardScaler()

def get_data(symbol, interval):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": 200,
        "apikey": TWELVE_DATA_API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "values" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["values"])
    df = df.iloc[::-1]
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }, inplace=True)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df["EMA"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df.dropna(inplace=True)
    return df

def market_structure(df):
    highs = df["High"].tail(12).values
    lows = df["Low"].tail(12).values
    if len(highs) < 2:
        return "ranging"
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"

def support_resistance(df):
    support = df["Low"].tail(12).min()
    resistance = df["High"].tail(12).max()
    return support, resistance

def create_features(df):
    last = df.iloc[-1]
    return np.array([[last["Close"], last["EMA"], last["RSI"], last["MACD"], last["MACD_signal"], last["ATR"]]])

def ensure_model_initialized(features):
    global model, scaler
    if not hasattr(model, "classes_"):
        scaler.fit(features)
        model.partial_fit(scaler.transform(features), [1], classes=[0, 1])

def analyze_timeframe(df):
    df = add_indicators(df)
    structure = market_structure(df)
    support, resistance = support_resistance(df)
    features = create_features(df)
    ensure_model_initialized(features)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0]
    buy_prob = float(prob[1])
    sell_prob = float(prob[0])
    price = df["Close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_signal"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    buy_score = 0
    sell_score = 0
    if structure == "bullish":
        buy_score += 1
    if structure == "bearish":
        sell_score += 1
    if rsi < 70:
        buy_score += 1
    if rsi > 30:
        sell_score += 1
    if macd > macd_signal:
        buy_score += 1
    if macd < macd_signal:
        sell_score += 1
    buy_score += buy_prob
    sell_score += sell_prob
    risk = atr
    rr = 2
    buy_sl = price - risk
    buy_tp = price + risk * rr
    sell_sl = price + risk
    sell_tp = price - risk * rr
    return {
        "buy_score": buy_score,
        "sell_score": sell_score,
        "buy_sl": buy_sl,
        "buy_tp": buy_tp,
        "sell_sl": sell_sl,
        "sell_tp": sell_tp,
        "price": price
    }

def update_model(df):
    df = add_indicators(df)
    if len(df) < 3:
        return
    features = create_features(df.iloc[:-1])
    future_close = df["Close"].iloc[-1]
    current_close = df["Close"].iloc[-2]
    target = 1 if future_close > current_close else 0
    scaler.partial_fit(features)
    model.partial_fit(scaler.transform(features), [target])
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for pair in PAIRS.keys():
        row.append(InlineKeyboardButton(pair, callback_data=pair))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select a pair:", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    pair_name = query.data
    symbol = PAIRS[pair_name]
    total_buy = 0
    total_sell = 0
    final_data = None
    for tf_name, tf_value in TIMEFRAMES.items():
        df = get_data(symbol, tf_value)
        if df.empty or len(df) < 50:
            continue
        result = analyze_timeframe(df)
        total_buy += result["buy_score"]
        total_sell += result["sell_score"]
        final_data = result
        update_model(df)
    if final_data is None:
        await query.edit_message_text("No data available.")
        return
    direction = "BUY" if total_buy > total_sell else "SELL"
    confidence = abs(total_buy - total_sell) / (total_buy + total_sell) * 100
    if direction == "BUY":
        text = f"{pair_name}\nDirection: BUY\nConfidence: {confidence:.2f}%\nEntry: {final_data['price']:.2f}\nSL: {final_data['buy_sl']:.2f}\nTP: {final_data['buy_tp']:.2f}"
    else:
        text = f"{pair_name}\nDirection: SELL\nConfidence: {confidence:.2f}%\nEntry: {final_data['price']:.2f}\nSL: {final_data['sell_sl']:.2f}\nTP: {final_data['sell_tp']:.2f}"
    await query.edit_message_text(text)

def main():
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.run_polling()

if __name__ == "__main__":
    main()
