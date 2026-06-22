from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import time

# 1. Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_engine.log"),  # Simpan log ke file
        logging.StreamHandler(),  # Tampilkan log di terminal
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Quant Engine")


def calculate_gbm_parameters(
    ticker: str = "BTC-USD", period: str = "1mo", interval: str = "1h"
):
    logger.info(
        f"Memulai kalkulasi GBM untuk {ticker} (Period: {period}, Interval: {interval})"
    )
    start_time = time.time()

    # 1. Tarik data historis
    logger.info(f"Mengunduh data dari Yahoo Finance...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        logger.error(f"Gagal mendapatkan data untuk {ticker}. Data kosong.")
        return None

    # Ambil harga penutupan (Close)
    close_prices = data["Close"]
    logger.info(f"Data berhasil diterima, jumlah bar: {len(close_prices)}")

    # 2. Hitung Log Returns
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    # 3. Hitung Parameter Statistik
    logger.info("Menghitung parameter statistik (Mean, Variance, StdDev)...")
    mean_return = np.mean(log_returns)
    variance = np.var(log_returns)
    std_dev = np.std(log_returns)

    # 4. Kalkulasi Drift dan Volatilitas
    drift = mean_return + (variance / 2.0)
    volatility = std_dev

    execution_time = time.time() - start_time
    logger.info(f"Kalkulasi selesai dalam {execution_time:.4f} detik.")

    return {
        "symbol": ticker,
        "metrics": {
            "drift_mu": (
                float(drift.iloc[0]) if isinstance(drift, pd.Series) else float(drift)
            ),
            "volatility_sigma": (
                float(volatility.iloc[0])
                if isinstance(volatility, pd.Series)
                else float(volatility)
            ),
        },
    }


@app.get("/api/v1/analyze/{symbol}")
def analyze_market(symbol: str):
    # 1. Mapping Simbol dari MT5 ke Yahoo Finance
    clean_symbol = (
        symbol.upper().replace("M", "").replace(".", "")
    )  # Membersihkan akhiran broker

    if "XAUUSD" in clean_symbol or "GOLD" in clean_symbol:
        yf_ticker = "GC=F"  # Ticker Emas
    elif "BTCUSD" in clean_symbol:
        yf_ticker = "BTC-USD"  # Ticker Bitcoin
    else:
        yf_ticker = clean_symbol  # Default

    logger.info(
        f"--- Request masuk: MT5({symbol}) diterjemahkan ke Yahoo({yf_ticker}) ---"
    )

    # 2. Proses kalkulasi menggunakan ticker yang sudah diterjemahkan
    result = calculate_gbm_parameters(ticker=yf_ticker)

    if not result:
        logger.error("Gagal memproses data!")
        return {"status": "error", "message": "Gagal mengambil data."}

    logger.info(f"Berhasil menganalisis {yf_ticker}")
    return {"status": "success", "data": result}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Quant Engine Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
