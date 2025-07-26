# النسخة المطورة من سكربت تحليل العملات سبوت (مع جني أرباح جزئي + فلترة + مؤشرات إضافية)
import sys
import asyncio
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.constants import ParseMode
import ccxt.async_support as ccxt
import os

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHANNEL_ID = os.environ["CHANNEL_ID"]
TIMEFRAME = os.environ.get("TIMEFRAME", "5m")
VOLUME_THRESHOLD = int(os.environ.get("VOLUME_THRESHOLD", 50))
TOP_N = int(os.environ.get("TOP_N", 200))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", 60))
PROFIT_TARGET_PERCENT = float(os.environ.get("PROFIT_TARGET_PERCENT", 10.0)) / 100
STOP_LOSS_PERCENT = float(os.environ.get("STOP_LOSS_PERCENT", 3.0)) / 100
DAILY_MAX_POSITIONS = int(os.environ.get("DAILY_MAX_POSITIONS", 5))
COOLDOWN_HOURS = int(os.environ.get("COOLDOWN_HOURS", 6))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", 14))
BOLLINGER_PERIOD = int(os.environ.get("BOLLINGER_PERIOD", 20))
BOLLINGER_STDDEV = int(os.environ.get("BOLLINGER_STDDEV", 2))

bot = Bot(token=TELEGRAM_TOKEN)
STABLECOINS = ["USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "EUR", "TRY", "GBP", "AUD"]

def load_json_file(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except:
        return {}

def save_json_file(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

sent_today = load_json_file("sent_today.json")
open_positions = load_json_file("open_positions.json")

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

async def get_top_symbols(exchange):
    await exchange.load_markets()
    tickers = await exchange.fetch_tickers()
    usdt_pairs = [
        symbol for symbol in tickers
        if symbol.endswith("/USDT") and not any(stable in symbol.split("/")[0] for stable in STABLECOINS)
    ]
    volumes = []
    for symbol in usdt_pairs:
        ticker = tickers.get(symbol)
        if ticker and ticker.get("quoteVolume"):
            volumes.append((symbol, ticker["quoteVolume"]))
    top = sorted(volumes, key=lambda x: x[1], reverse=True)[:TOP_N]
    return [x[0] for x in top]

async def analyze_symbol(exchange, symbol):
    now = datetime.now(timezone.utc)
    today = now.strftime('%Y-%m-%d')
    last_sent = sent_today.get(symbol)

    # الحد الأقصى اليومي
    if sum(1 for v in sent_today.values() if v["time"].startswith(today)) >= DAILY_MAX_POSITIONS:
        return

    # فترة التهدئة (Cooldown)
    if last_sent:
        sent_time = datetime.fromisoformat(last_sent["time"])
        if (now - sent_time) < timedelta(hours=COOLDOWN_HOURS):
            return

    try:
        df = pd.DataFrame(await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100),
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['ema100'] = df['close'].ewm(span=100).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        df['macd'], df['macd_signal'] = calculate_macd(df['close'])
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        df['bb_upper'], df['bb_mid'], df['bb_lower'] = calculate_bollinger_bands(df['close'], BOLLINGER_PERIOD, BOLLINGER_STDDEV)

        macd_cross = df['macd'].iloc[-2] < df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
        trend_up = df['ema100'].iloc[-1] > df['ema200'].iloc[-1]
        rsi_ok = 30 < df['rsi'].iloc[-1] < 70
        bollinger_ok = df['close'].iloc[-1] > df['bb_mid'].iloc[-1]

        price = df['close'].iloc[-1]

        if trend_up and macd_cross and rsi_ok and bollinger_ok:
            message_id = await send_signal(symbol, price)
            sent_today[symbol] = {"price": price, "time": now.isoformat()}
            open_positions[symbol] = {
                "entry": price,
                "targets": [round(price * (1 + PROFIT_TARGET_PERCENT * i), 4) for i in range(1, 6)],
                "stop": round(price * (1 - STOP_LOSS_PERCENT), 4),
                "hit": [],
                "message_id": message_id,
                "partial_sell_flags": [],
                "stopped": False
            }
            save_json_file("sent_today.json", sent_today)
            save_json_file("open_positions.json", open_positions)

    except Exception as e:
        print(f"[{datetime.utcnow()}] [!] فشل تحليل {symbol}: {e}")

async def send_signal(symbol, entry_price):
    targets = [round(entry_price * (1 + PROFIT_TARGET_PERCENT * i), 4) for i in range(1, 6)]
    stop = round(entry_price * (1 - STOP_LOSS_PERCENT), 4)
    profit_targets = [f"• {i+1}⃣ {t} (+{round((t-entry_price)/entry_price*100, 2)}%)" for i, t in enumerate(targets)]

    message = f"""
🚨 <b>صفقة شراء (Spot)</b>
#<b>{symbol.replace('/USDT','')}</b>

🎯 <b>سعر الدخول:</b> {entry_price:.4f}
📊 <b>الأهداف:</b>
{chr(10).join(profit_targets)}

🛑 <b>وقف الخسارة:</b> {stop}
🕓 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
    """
    msg = await bot.send_message(chat_id=CHANNEL_ID, text=message, parse_mode=ParseMode.HTML)
    return msg.message_id

async def update_signal_message(symbol, price, data):
    profit_targets = []
    for i, target in enumerate(data["targets"]):
        hit = "✅" if i in data["hit"] else "•"
        percent = round((target - data["entry"]) / data["entry"] * 100, 2)
        profit_targets.append(f"{hit} {i+1}⃣ {target} (+{percent}%)")

    stop_note = f"❌ <b>تم ضرب وقف الخسارة عند السعر {price:.4f}</b>" if data.get("stopped") else ""

    message = f"""
📌 <b>صفقة سبوت شراء (Spot Long)</b>
#<b>{symbol.replace('/USDT','')}</b>

🎯 <b>سعر الدخول:</b> {data['entry']:.4f}
📊 <b>الأهداف:</b>
{chr(10).join(profit_targets)}

🛑 <b>وقف الخسارة:</b> {data['stop']}
{stop_note}
💰 <b>السعر الحالي:</b> {price:.4f}
⏱️ <i>آخر تحديث: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
    """
    try:
        await bot.edit_message_text(chat_id=CHANNEL_ID, message_id=data["message_id"], text=message, parse_mode=ParseMode.HTML)
    except Exception as e:
        print(f"[!] فشل تحديث رسالة {symbol}: {e}")

async def monitor_targets(exchange):
    while True:
        for symbol in list(open_positions.keys()):
            try:
                data = open_positions[symbol]
                price = (await exchange.fetch_ticker(symbol))["last"]
                updated = False

                for i, target in enumerate(data["targets"]):
                    if i in data["hit"]:
                        continue
                    if price >= target:
                        data["hit"].append(i)
                        updated = True
                        if i in [0, 1, 2] and i not in data.get("partial_sell_flags", []):
                            await bot.send_message(
                                chat_id=CHANNEL_ID,
                                text=f"📤 <b>اقتراح بيع 20%</b> من صفقة {symbol} بعد تحقق الهدف {i+1}.",
                                parse_mode=ParseMode.HTML
                            )
                            data.setdefault("partial_sell_flags", []).append(i)

                if price <= data["stop"] and not data.get("stopped"):
                    data["stopped"] = True
                    updated = True
                    await bot.send_message(
                        chat_id=CHANNEL_ID,
                        text=f"❌ <b>تم ضرب وقف الخسارة لصفقة {symbol} عند السعر {price:.4f}</b>",
                        parse_mode=ParseMode.HTML
                    )

                if updated:
                    await update_signal_message(symbol, price, data)
                    if data.get("stopped"):
                        del open_positions[symbol]
                    save_json_file("open_positions.json", open_positions)
            except Exception as e:
                print(f"[!] خطأ في مراقبة {symbol}: {e}")

        await asyncio.sleep(CHECK_INTERVAL)

async def main_loop():
    exchange = ccxt.binance()
    asyncio.create_task(monitor_targets(exchange))

    while True:
        try:
            symbols = await get_top_symbols(exchange)
            tasks = [analyze_symbol(exchange, s) for s in symbols]
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"[X] ❌ خطأ عام: {e}")
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main_loop())