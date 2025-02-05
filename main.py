import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta
from textblob import TextBlob
import ccxt
from datetime import datetime

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = ccxt.binance()
DAYS_TO_PREDICT = 7

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Crypto Prediction Bot\nUse /predict [coin_id]")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].lower() if context.args else 'bitcoin'
    try:
        data = get_crypto_data(symbol)
        real_time_data = get_real_time_data(symbol)
        sentiment = get_sentiment(symbol)
        
        results = []
        results.append(technical_analysis_prediction(data))
        results.append(lstm_prediction(data))
        results.append(prophet_prediction(data))
        
        best_strategy = max(results, key=lambda x: x['confidence'])
        risk_reward = calculate_risk_reward(data, best_strategy['prediction'])
        
        response = (
            f"📈 **{symbol.upper()} {DAYS_TO_PREDICT}-Day Predictions**\n\n"
            f"• Real-Time Price: ${real_time_data['price']:.2f}\n"
            f"• Volume (24h): {real_time_data['volume']:.2f}\n"
            f"• Sentiment: {sentiment['sentiment']:.2f} ({sentiment['tweet_count']} tweets)\n\n"
        )
        
        for result in results:
            response += (
                f"• {result['strategy']}:\n"
                f"  Price: ${result['prediction']:.2f}\n"
                f"  Confidence: {result['confidence']*100:.1f}%\n"
                f"  Volatility: {result['volatility']*100:.1f}%\n"
            )
            if 'signals' in result:
                response += f"  Signals: {', '.join(result['signals'])}\n"
        
        response += (
            f"\n🌟 **Recommended Strategy**: {best_strategy['strategy']} "
            f"(Confidence: {best_strategy['confidence']*100:.1f}%)\n"
            f"📊 **Risk/Reward Ratio**: {risk_reward:.2f}\n"
        )
        
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

# Main Function
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    
    application.run_polling()

if __name__ == '__main__':
    main()