from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC/XAU Quant Engine")

# --- SISTEM MEMORI CACHE GLOBAL ---
cache_data = {}
CACHE_DURATION_SEC = 900  # Cache disimpan selama 15 menit (900 detik)


def calculate_gbm_parameters(ticker: str, period: str = "1mo", interval: str = "1h"):
    current_time = time.time()

    # Cek apakah data sudah ada di cache dan belum kedaluwarsa
    if ticker in cache_data:
        time_elapsed = current_time - cache_data[ticker]["timestamp"]
        if time_elapsed < CACHE_DURATION_SEC:
            logger.info(
                f"[CACHE HIT] Menggunakan memori lokal untuk {ticker} (Sisa waktu cache: {int(CACHE_DURATION_SEC - time_elapsed)} detik)"
            )
            return cache_data[ticker]["result"]

    # Jika tidak ada di cache / sudah kedaluwarsa, ambil data baru dari Yahoo
    logger.info(
        f"[CACHE MISS] Mengunduh data segar dari Yahoo Finance untuk {ticker}..."
    )
    start_time = time.time()

    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        logger.error(f"Gagal mendapatkan data dari Yahoo untuk {ticker}")
        return None

    close_prices = data["Close"]
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    mean_return = np.mean(log_returns)
    variance = np.var(log_returns)
    std_dev = np.std(log_returns)

    drift = mean_return + (variance / 2.0)
    volatility = std_dev

    final_result = {
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

    # Simpan hasil kalkulasi ke dalam cache
    cache_data[ticker] = {"timestamp": current_time, "result": final_result}

    logger.info(
        f"Kalkulasi baru selesai dalam {time.time() - start_time:.4f} detik. Disimpan ke cache."
    )
    return final_result


@app.get("/api/v1/analyze/{symbol}")
def analyze_market(symbol: str):
    clean_symbol = symbol.upper().replace("M", "").replace(".", "")

    if "XAUUSD" in clean_symbol or "GOLD" in clean_symbol:
        yf_ticker = "GC=F"
    elif "BTCUSD" in clean_symbol:
        yf_ticker = "BTC-USD"
    else:
        yf_ticker = clean_symbol

    result = calculate_gbm_parameters(ticker=yf_ticker)

    if not result:
        return {"status": "error", "message": "Gagal memproses data."}

    return {"status": "success", "data": result}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Quant Engine Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
